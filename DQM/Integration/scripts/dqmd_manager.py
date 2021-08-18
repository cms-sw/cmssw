#!/usr/bin/env python3

from __future__ import print_function
import sys
import subprocess


COLOR_PURPLE = '\033[95m'
COLOR_BLUE = '\033[94m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_RED = '\033[91m'
COLOR_DEF = '\033[0m'

# This variable is used only when the daemon is installed/updated 
# by hand and needs to be explicitly specified by the user.
rpm_path = '' # '/nfshome0/smorovic/gcc481/dqm/hltd-1.5.0-2.x86_64.rpm'

def rpm_install(machine):
    return 'sudo rpm --install {0}'.format(rpm_path)

def rpm_update(machine):
    return 'sudo rpm -Uhv --force {0}'.format(rpm_path)


machines = { 'bu' : ['dqm-c2d07-22', 'bu-c2f13-31-01', 'bu-c2f13-29-01'],
             'dev' : ['dqm-c2d07-21', 'dqm-c2d07-22', 'dqm-c2d07-23', 'dqm-c2d07-24', 'dqm-c2d07-25', 'dqm-c2d07-26', 'dqm-c2d07-27'],
             'dev_current' : ['dqm-c2d07-22', 'dqm-c2d07-23'],
             'ed' : ['bu-c2f13-29-01', 'fu-c2f13-41-01', 'fu-c2f13-41-02', 'fu-c2f13-41-03', 'fu-c2f13-41-04'],
             'ed_current' : ['bu-c2f13-29-01', 'fu-c2f13-41-03'],
             'prod' : ['bu-c2f13-31-01', 'fu-c2f13-39-01', 'fu-c2f13-39-02', 'fu-c2f13-39-03', 'fu-c2f13-39-04'],
             'prod_current' : ['bu-c2f13-31-01', 'fu-c2f13-39-04']
           }


actions = { 'rpm_install' : rpm_install,  
            'rpm_update' : rpm_update,
            'rpm_install_status' : 'rpm -qa hltd',
            'rpm_remove' : 'sudo rpm --erase hltd',
            'daemon_status' : 'sudo /sbin/service hltd status',
            'daemon_start' : 'sudo /sbin/service hltd start',
            'daemon_stop' : 'sudo /sbin/service hltd stop',
            'daemon_stop-light' : 'sudo /sbin/service hltd stop-light',
            'daemon_restart' : 'sudo /sbin/service hltd restart'}


def usage():
    print('Usage: ' + sys.argv[0] + ' MACHINES ACTIONS')

    print('\tMACHINES:')
    for target in machines.keys():
        print('\t\t' + target + ': ' + ', '.join(machines[target]))

    print('\tACTIONS:')
    for action in actions.keys():
        print('\t\t' + action)


def info(info=None):
    if None != info:
        print(COLOR_BLUE + '***************************** ' + info + ' *****************************' + COLOR_DEF)
    else:
        print(COLOR_BLUE + '*********************************************************************************' + COLOR_DEF)


def exec_func(machine, action):
    info('Machine: ' + machine)
    call_list = []
    call_list.append('ssh')
    call_list.append(machine)

    if hasattr(action, '__call__'):
        call_list.append(action(machine))
    else:
        call_list.append(action)

    # print(call_list) # DEBUG_CODE
    subprocess.call(call_list, stderr=subprocess.STDOUT)

    info()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
        exit(1)

    targets = machines.get(sys.argv[1])
    if targets == None:
        print('Wrong target machines')
        exit(2)

    action = actions.get(sys.argv[2])
    if action == None:
        print('Wrong action')
        exit(3)

    for target in targets:
        exec_func(target, action)

