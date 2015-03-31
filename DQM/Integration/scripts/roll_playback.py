#!/usr/bin/env python2

# TODO list
# - handle the situation where no .jsn of data files are found in the source directory in a better way
# - automatically determine from which LS to start (currently this is hard-coded to 1)
# - when dealing with file I/O use the python "file scope"

import os
import sys
import time
import re
import shutil
import json

dat_source   = '/fff/ramdisk/playback_files/run233238/'
pb_source    = '/fff/ramdisk/playback_files/run233238/'
calib_source = '/fff/ramdisk/playback_files/run233238/'

destination = '/fff/ramdisk'
lumi_len = 23 # in seconds
run_padding = 6
lumi_padding = 4
files_copied_buffer_len = 60 # the number of file to keep in the ramdisk
run_switch_interval = 90 # in seconds

lumi_skip_length = 10 

file_types = { 'general_files': {'extension':'.dat', 're_pattern':r'run([0-9]+)_ls([0-9]+)_streamDQM_mrg-[A-Za-z0-9-]+\.jsn'},
               'hlt_pb_files':  {'extension':'.pb',  're_pattern':r'run([0-9]+)_ls([0-9]+)_streamDQMHistograms_mrg-[A-Za-z0-9-]+\.jsn'},
               'calib_files':   {'extension':'.dat', 're_pattern':r'run([0-9]+)_ls([0-9]+)_streamDQMCalibration_mrg-[A-Za-z0-9-]+\.jsn'}, }


def sanity_check(source, file_info):
    jsn_files = []
    data_files = []
    run_number = None

    # find the json files that match the given pattern
    files = os.listdir(source)
    jsn_pattern = re.compile(file_info['re_pattern'])
    jsn_files = sorted(filter(lambda x: jsn_pattern.match(x), files))

    # check if the data files exist
    jsn_files_tobe_removed = []
    for jsn_file in jsn_files:
        data_file = jsn_file.replace('.jsn', file_info['extension'])
        if os.path.exists(source + '/' + data_file):
            data_files.append(data_file)
        else:
            print('The data file {0} does NOT exist! Removing the corresponding json file.'.format(data_file))
            jsn_files_tobe_removed.append(jsn_file)

    # remove the json files that don't have corresponding data file
    jsn_files = [x for x in jsn_files if x not in jsn_files_tobe_removed]

    run_number = int(jsn_pattern.match(jsn_files[0]).group(1))
    # check for run_number consistency
    for i in range(1,len(jsn_files)):
        run_number_current = int(jsn_pattern.match(jsn_files[i]).group(1))
        if run_number_current != run_number:
            print('Non consistent run numbers: "{0}" - expected, "{1}" - found'.format(run_nummber, run_nummber_current))
            print('\t "{0}" - will be used as a run number'.format(run_nummber))

    return True, jsn_files, data_files, run_number


def copy_next_lumi(jsn_file, file, run_number, current_lumi, source, destination, copy_file=True):
    index = current_lumi % len(jsn_file)

    # copy the file
    input_fn = source + '/' + file
    output_fn = file
    run_start = output_fn.find('run') + 3
    output_fn = output_fn[:run_start] + str(run_number).zfill(run_padding) + output_fn[run_start + run_padding:]
    lumi_start = output_fn.find('ls') + 2
    output_fn = destination + '/' + output_fn[:lumi_start] + str(current_lumi).zfill(lumi_padding) + output_fn[lumi_start + lumi_padding:]
    if copy_file:
        os.link(input_fn, output_fn) # instead of copying the file create a hard link
        print(input_fn + ' -> ' + output_fn)

    # load the original json contents
    input_jsn_fn = source + '/' + jsn_file
    input_jsn = open(input_jsn_fn, 'r')
    jsn_data = json.load(input_jsn)
    input_jsn.close()

    # generate the output jsn file name
    output_jsn_fn = jsn_file
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

    return (output_jsn_fn, output_fn) if copy_file else (output_jsn_fn, )


if __name__ == '__main__':
    dat_dir_ok, dat_jsn_files, dat_files, run_number = sanity_check(dat_source, file_types['general_files'])
    pb_dir_ok, pb_jsn_files, pb_files, pb_run_number = sanity_check(pb_source, file_types['hlt_pb_files'])
    calib_dir_ok, calib_jsn_files, calib_files, calib_run_number = sanity_check(calib_source, file_types['calib_files'])

    if dat_dir_ok and pb_dir_ok and calib_dir_ok:
        if (run_number != pb_run_number) or (run_number != calib_run_number):
            print('The DAT run number differs from the PB or Calibration run number.')
            print('"{0}" is going to be used as a run number. \n'.format(run_number))

        run_length = len(dat_jsn_files)
        lumi_skip_at = None
        copy_file = True
        if run_length > 25:
            lumi_skip_at = run_length/10

        files_copied = []

        while True:
            global_runfile = destination + '/' + '.run{0}.global'.format(str(run_number).zfill(run_padding))
            gf = open(global_runfile, 'w')
            gf.write('run_type = cosmic_run')
            gf.close()

            output_dir = destination + '/' + 'run' + str(run_number).zfill(run_padding)
            os.mkdir(output_dir)

            time.sleep(1) # a hack in order python inotify to work correctly

            current_lumi = 1
            for i in range(len(dat_jsn_files)):
                files_copied += copy_next_lumi(dat_jsn_files[i], dat_files[i], run_number, current_lumi, dat_source, output_dir, copy_file)

                j = i%len(pb_jsn_files)
                files_copied += copy_next_lumi(pb_jsn_files[j], pb_files[j], run_number, current_lumi, pb_source, output_dir, copy_file)

                k = i%len(calib_jsn_files)
                files_copied += copy_next_lumi(calib_jsn_files[k], calib_files[k], run_number, current_lumi, calib_source, output_dir, copy_file)

                if not lumi_skip_at or (current_lumi != lumi_skip_at): current_lumi += 1
                else: current_lumi += lumi_skip_length

                if not lumi_skip_at or (current_lumi < 2*lumi_skip_at) or (current_lumi > 2*lumi_skip_at+lumi_skip_length): copy_file = True
                else: copy_file = False

                time.sleep(lumi_len)

                # clear some of the old files
                while files_copied_buffer_len < len(files_copied):
                    os.remove(files_copied.pop(0))

                print('')

            EoRfile = output_dir + '/' + 'run' + str(run_number).zfill(run_padding) + '_ls0000_EoR.jsn' # create the EoR file
            open(EoRfile, 'w').close()
            run_number += 1
            print('\n\n')
            time.sleep(run_switch_interval)

