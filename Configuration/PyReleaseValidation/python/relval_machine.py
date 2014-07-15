from  Configuration.PyReleaseValidation.relval_steps import Matrix, InputInfo, Steps
import os
import json
import collections


workflows = Matrix()
steps = Steps()


def get_json_files():
    cwd = os.path.join(os.getcwd(), "json_data")
    if not os.path.exists(cwd):
        return []

    json_files = []
    for f in os.listdir(cwd):
        full_path = os.path.join(cwd, f)
        if os.path.isfile(full_path) and f.endswith(".json"):
            json_files.append(full_path)
    return json_files


def fix_run(run):
    runs = run.replace(" ", "").replace("[", "").replace("]", "").split(",")
    int_runs = []
    for item in runs:
        if item.isdigit():
            int_runs.append(int(item))
        else:
            print "WARNING: run is in bad format: {0}".format(run)
    return int_runs

def convert_keys_to_string(dictionary):
    """ Recursively converts dictionary keys to strings.
        Utility to help deal with unicode keys in dictionaries created from json requests.
        In order to pass dict to function as **kwarg we should transform key/value to str.
    """
    if isinstance(dictionary, basestring):
        return str(dictionary)
    elif isinstance(dictionary, collections.Mapping):
        return dict(map(convert_keys_to_string, dictionary.iteritems()))
    elif isinstance(dictionary, collections.Iterable):
        return type(dictionary)(map(convert_keys_to_string, dictionary))
    else:
        return dictionary

def load_steps_and_workflows():
    data_files = get_json_files()
    for index, data_file in enumerate(data_files):
        with open(data_file, "r") as f:
            data = json.load(f)
            data = convert_keys_to_string(data)
            label = data["label"]
            steps_names = []
            for step_name, step in data["steps"].items():
                steps_names.append((step_name, step["sequence_number"]))
                if step_name in steps:
                    continue  # this step was inserted already

                # inputInfo case
                if "inputInfo" in step:
                    input_info = step["inputInfo"]
                    if "run" in input_info:
                        input_info["run"] = fix_run(input_info["run"])

                    steps[step_name] = {
                        'INPUT': InputInfo(**input_info)
                    }
                # step with parameters
                elif "parameters" in step:
                    steps[step_name] = step["parameters"]
                else:
                    raise Exception("Wrong step format in {0} file".format(data_file))

            sorted_steps = sorted(steps_names, key=lambda step: step[1]) # sort steps by sequence number
            sorted_steps_names = [step_name[0] for step_name in sorted_steps]  # filter only step names

            workflows[1000000.0 + 0.1*index] = [label, sorted_steps_names]


load_steps_and_workflows()

