#!/usr/bin/env python2

# TODO: automatically determine from which LS to start (currently this is hard-coded to 1)


import os
import sys
import time
import re
import shutil
import json


dat_source = '/fff/ramdisk/playback_files/run224380'
pb_source = '/fff/ramdisk/playback_files/run225044_pb'
destination = '/fff/ramdisk'
lumi_len = 23 # in seconds
run_padding = 6
lumi_padding = 4
files_copied_buffer_len = 20 # the number of file to keep in the destination directory


def dat_sanity_check(dat_source):
    dat_jsn_files = []
    dat_files = []
    dat_run_number = None

    # find the dat json files
    files = os.listdir(dat_source)
    dat_jsn_pattern = re.compile(r'run([0-9]+)_ls([0-9]+)_streamDQM_StorageManager.jsn')
    dat_jsn_files = sorted(filter(lambda x: dat_jsn_pattern.match(x), files))
    if len(dat_jsn_files) < 1:
        print('No dat json files are found in "{0}"'.format(dat_source))
        return False, dat_jsn_files, dat_files, dat_run_number

    # check if the dat files exist
    jsn_files_tobe_removed = []
    for jsn_file in dat_jsn_files:
        dat_file = jsn_file.replace('.jsn','.dat')
        if not os.path.exists(dat_source + '/' + dat_file):
            print('The dat file {0} does NOT exist! Removing the corresponding json file.'.format(dat_file))
            jsn_files_tobe_removed.append(jsn_file)

    # remove the json files that don't have corresponding dat file
    dat_jsn_files = [x for x in dat_jsn_files if x not in jsn_files_tobe_removed]

    # create a list of dat files
    dat_files = map(lambda x: x.replace('.jsn','.dat'), dat_jsn_files)


    dat_run_number = int(dat_jsn_pattern.match(dat_jsn_files[0]).group(1))
    # check for run_number consistency
    for i in range(1,len(dat_jsn_files)):
        run_number_current = int(dat_jsn_pattern.match(dat_jsn_files[i]).group(1))
        if run_number_current != dat_run_number:
            print('Non consistent run numbers: "{0}" - expected, "{1}" - found'.format(run_nummber, run_nummber_current))
            print('\t "{0}" - will be used as a run number'.format(run_nummber))

    return True, dat_jsn_files, dat_files, dat_run_number


def pb_sanity_check(pb_source):
    pb_jsn_files = []
    pb_files = []
    pb_run_number = None

    # find the pb json files
    files = os.listdir(pb_source)
    pb_jsn_pattern = re.compile(r'run([0-9]+)_ls([0-9]+)_streamDQMHistograms_StorageManager.jsn')
    pb_jsn_files = sorted(filter(lambda x: pb_jsn_pattern.match(x), files))

    # check if the pb files exist
    jsn_files_tobe_removed = []
    for jsn_file in pb_jsn_files:
        pb_file = jsn_file.replace('.jsn','.pb')
        if not os.path.exists(pb_source + '/' + pb_file):
            print('The pb file {0} does NOT exist! Removing the corresponding json file.'.format(pb_file))
            jsn_files_tobe_removed.append(jsn_file)

    # remove the json files that don't have corresponding pb file
    pb_jsn_files = [x for x in pb_jsn_files if x not in jsn_files_tobe_removed]

    if len(pb_jsn_files) < 1:
        print('No pb json files are found in "{0}"'.format(pb_source))
        return False, pb_jsn_files, pb_files, pb_run_number

    # create a list of pb files
    pb_files = map(lambda x: x.replace('.jsn','.pb'), pb_jsn_files)

    pb_run_number = int(pb_jsn_pattern.match(pb_jsn_files[0]).group(1))
    # check for run_number consistency
    for i in range(1,len(pb_jsn_files)):
        run_number_current = int(pb_jsn_pattern.match(pb_jsn_files[i]).group(1))
        if run_number_current != pb_run_number:
            print('Non consistent run numbers: "{0}" - expected, "{1}" - found'.format(run_nummber, run_nummber_current))
            print('\t "{0}" - will be used as a run number'.format(run_nummber))

    return True, pb_jsn_files, pb_files, pb_run_number


def copy_next_lumi(jsn_files, files, run_number, current_lumi, source, destination):
    assert(len(jsn_files) == len(files))

    index = current_lumi % len(jsn_files)

    # copy the file
    input_fn = source + '/' + files[index]
    output_fn = files[index]
    run_start = output_fn.find('run') + 3
    output_fn = output_fn[:run_start] + str(run_number).zfill(run_padding) + output_fn[run_start + run_padding:]
    lumi_start = output_fn.find('ls') + 2
    output_fn = destination + '/' + output_fn[:lumi_start] + str(current_lumi).zfill(lumi_padding) + output_fn[lumi_start + lumi_padding:]
    os.link(input_fn, output_fn) # instead of copying the file create a hard link
    print(input_fn + ' -> ' + output_fn)

    # modyfy and copy the json file
    input_jsn_fn = source + '/' + jsn_files[index]
    input_jsn = open(input_jsn_fn, 'r')
    jsn_data = json.load(input_jsn)
    input_jsn.close()

    # generate the output jsn file name
    output_jsn_fn = jsn_files[index]
    run_start = output_jsn_fn.find('run') + 3
    output_jsn_fn = output_jsn_fn[:run_start] + str(run_number).zfill(run_padding) + output_jsn_fn[run_start + run_padding:]
    lumi_start = output_jsn_fn.find('ls') + 2
    output_jsn_fn = destination + '/' + output_jsn_fn[:lumi_start] + str(current_lumi).zfill(lumi_padding) + output_jsn_fn[lumi_start + lumi_padding:]

    # modify the json file contents
    jsn_data['data'][3] = output_fn[output_fn.rfind('/')+1:]

    # create the outpuf jsn file
    output_jsn = open(output_jsn_fn, 'w')
    output_jsn.write(json.dumps(jsn_data, indent=4))
    output_jsn.close()

    print(input_jsn_fn + ' -> ' + output_jsn_fn)

    return output_jsn_fn, output_fn



if __name__ == '__main__':
    dat_dir_ok, dat_jsn_files, dat_files, dat_run_number = dat_sanity_check(dat_source)
    pb_dir_ok, pb_jsn_files, pb_files, pb_run_number = pb_sanity_check(pb_source)

    if dat_dir_ok and pb_dir_ok:
        run_number = int(dat_run_number)
        if run_number != int(pb_run_number):
            print('The dat run number "{0}" differs from the PB run number "{1}".'.format(run_number, pb_run_number))
            print('"{0}" is going to be used as a run number.'.format(run_number))


        output_dir = destination + '/' + 'run' + str(dat_run_number).zfill(run_padding)
        if not os.path.exists(output_dir): os.mkdir(output_dir)

        time.sleep(1) # a hack in order python inotify to work correctly

        current_lumi = 1
        files_copied = []
        while True:
            files_copied += copy_next_lumi(dat_jsn_files, dat_files, run_number, current_lumi, dat_source, output_dir)

            files_copied += copy_next_lumi(pb_jsn_files, pb_files, run_number, current_lumi, pb_source, output_dir)

            print('******************************************************************************************')

            while files_copied_buffer_len < len(files_copied):
                os.remove(files_copied.pop(0))

            current_lumi += 1
            time.sleep(lumi_len)
