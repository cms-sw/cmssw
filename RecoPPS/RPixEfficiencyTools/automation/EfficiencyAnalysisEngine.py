from weakref import WeakKeyDictionary
import automation_control as ctrl
import argparse
import subprocess
import enum 
import logging
from typing import Any, Type, Union
from os import listdir, walk, environ
from os.path import isfile, join

logger = logging.getLogger("EfficiencyAnalysisLogger")
logger.setLevel(logging.DEBUG)

ch = logging.FileHandler("EfficiencyAnalysisEngine.log")
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

campaign=environ.get("CAMPAIGN")
workflow=environ.get("WORKFLOW")
dataset=environ.get("DATASET")
proxy=environ.get("PROXY")

template_for_first_module = "CrabConfigTemplateForFirstModule.py"
template_for_second_module = "CrabConfigTemplateForSecondModule.py"



@ctrl.define_status_enum
class TaskStatusEnum(enum.Enum):
    """
        Class to encode enum tasks statuses for the purpouse of this automation workflow
    """
    initialized = enum.auto(),
    duringFirstWorker = enum.auto(),
    waitingForFirstWorkerTransfer= enum.auto(),
    duringFirstHarvester = enum.auto(),
    waitingForFirstHarvester = enum.auto(),
    afterFirstHarvester = enum.auto(),
    duringSecondWorker = enum.auto(),
    waitingForSecondWorkerTransfer= enum.auto(),
    duringSecondHarvester = enum.auto(),
    waitingForSecondHarvester = enum.auto(),
    afterSecondHarvester = enum.auto(),
    done = enum.auto()
    

@ctrl.decorate_with_enum(TaskStatusEnum)
class TaskStatus:
    loop_id = 0.0    
    condor_job_id = 0

def get_tasks_numbers_list(tasks_list_path):
    with open(tasks_list_path) as tasks_list_path:
        tasks_list_data = tasks_list_path.read()
        tasks_list_data = tasks_list_data.replace(" ", "")
        tasks_list = tasks_list_data.split(",")
    return tasks_list


def prepare_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=
    """This is a script to run PPS Efficiency Analysis automation workflow""", formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-t', '--tasks_list', dest='tasks_list_path', help='path to file containing list of data periods', required=True)   
    return parser


def get_runs_range(data_period):
    """MOCKED"""
    return '317080'


def process_new_tasks(tasks_list_path, task_controller):
    tasks_list = get_tasks_numbers_list(tasks_list_path)
    tasks_list = set(tasks_list)
    tasks_in_database = task_controller.getAllTasks().get_points()
    tasks_in_database = set(map(lambda x: x['dataPeriod'], tasks_in_database))
    tasks_not_submited_yet = tasks_list-tasks_in_database
    if tasks_not_submited_yet:
        task_controller.submitTasks(tasks_not_submited_yet)
   

def flatten(t):
     return [item for sublist in t for item in sublist]


def submit_task_to_crab(campaign, workflow, data_period, dataset, template, proxy):
    result = ctrl.submit_task_to_crab(campaign, workflow, data_period, get_runs_range(data_period), template, dataset, proxy)

    return result


def set_status_after_first_worker_submission(task_status, operation_result):
    task_status.duringFirstWorker=1
    task_status.initialized=0
    task_status.loop_id+=1
    return task_status


executable = """
                cmsRun /afs/cern.ch/user/e/ecalgit/CMSSW_11_3_2/src/RecoPPS/RPixEfficiencyTools/python/EfficiencyAnalysisDQMHarvester_cfg.py 
                inputFileName=<input_files> 
                outputDirectoryPath=<output_dir>
                campaign=<campaign>
                workflow=<workflow>
                dataPeriod=<data_period>
             """

storage_path = "/eos/user/l/lkita"

def aggregate_files(path: str) -> str:
    if path[-1] != '/':
        path+='/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    if files:
        prefix = files[0][:files[0].rfind("_")] 
        if list(filter(lambda x: not x.startswith(prefix), files)):
            raise Exception("In: "+path+" directory there are files with different prefixes")

    files =  list(map(lambda x: path+x, files))
    return ",".join(files)


def submit_task_to_condor(campaign, workflow, data_period):
    global executable
    input_files_path = storage_path+"/"+"/".join([campaign, workflow, data_period])
    dirs_iterator = walk(input_files_path)
    for _ in range(5):
        dir_name = next(dirs_iterator)
        input_files_path = dir_name[0]
    
    executable = executable.replace("<input_files>", aggregate_files(input_files_path) )
    output_dir = "/afs/cern.ch/user/e/ecalgit/CMSSW_11_3_2/src/RecoPPS/RPixEfficiencyTools/OutputFiles/"+"/".join([campaign, workflow, data_period])
    executable = executable.replace("<output_dir>", output_dir)
    executable = executable.replace("<campaign>", campaign)
    executable = executable.replace("<workflow>", workflow)
    executable = executable.replace("<data_period>", data_period)
    executable = executable.replace("\n", " ")
    
    return ctrl.submit_task_to_condor(campaign, workflow, data_period, executable)


def set_status_during_first_harvester(task_status, cluster_id):
    task_status.waitingForFirstHarvester=1
    task_status.duringFirstHarvester=0
    task_status.condor_job_id=cluster_id
    return task_status


def set_status_during_second_harvester(task_status, cluster_id):
    task_status.waitingForSecondHarvester=1
    task_status.duringSecondHarvester=0
    task_status.condor_job_id=cluster_id
    return task_status


def wait_for_condor(campaign, workflow, data_period):
    task_controller = ctrl.TaskCtrl.TaskControl(campaign=campaign, workflow=workflow, TaskStatusClass=TaskStatus)
    last_task_status = task_controller.getLastTask(data_period=data_period)
    cluster_id = int(last_task_status.get("condor_job_id"))
    return ctrl.check_if_condor_task_is_finished(cluster_id)


TRANSITIONS_DICT = {
                        'initialized': (submit_task_to_crab, 0, set_status_after_first_worker_submission, [dataset, template_for_first_module, proxy] ),
                        'duringFirstWorker': (ctrl.check_if_crab_task_is_finished, True, TaskStatus.waitingForFirstWorkerTransfer, [proxy]),
                        'waitingForFirstWorkerTransfer': (ctrl.is_crab_output_already_transfered, True, TaskStatus.duringFirstHarvester, [proxy]),
                        'duringFirstHarvester': (submit_task_to_condor, ctrl.AnyInt, set_status_during_first_harvester, []),
                        'waitingForFirstHarvester': (wait_for_condor, True, TaskStatus.afterFirstHarvester, []),
                        'afterFirstHarvester': (submit_task_to_crab, 0, TaskStatus.duringSecondWorker, [dataset, template_for_second_module] ),
                        'duringSecondWorker': (ctrl.check_if_crab_task_is_finished, True, TaskStatus.waitingForSecondWorkerTransfer, [proxy]),
                        'waitingForSecondWorkerTransfer': (ctrl.is_crab_output_already_transfered, True, TaskStatus.duringSecondHarvester, [proxy]),
                        'duringSecondHarvester': (submit_task_to_condor, ctrl.AnyInt, set_status_during_second_harvester, []),
                        'waitingForSecondHarvester': (wait_for_condor, True, TaskStatus.done, [])
                   } 


    
if __name__ == '__main__':
    parser = prepare_parser()
    opts = parser.parse_args()
    task_controller = ctrl.TaskCtrl.TaskControl(campaign=campaign, workflow=workflow, TaskStatusClass=TaskStatus)
    process_new_tasks(opts.tasks_list_path, task_controller)
    finite_state_machine = ctrl.FiniteStateMachine(TRANSITIONS_DICT)
    finite_state_machine.process_tasks(task_controller, TaskStatusClass=TaskStatus)
    
    
    