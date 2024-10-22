#!/bin/env python3

import os
import sys
import glob
import argparse
import subprocess
from threading import Thread

COMPARISON_RESULTS = []

def collect_and_compare_files(base_dir, pr_dir, output_dir, num_procs, pr_number, test_number, release_format):
    files = get_file_pairs(base_dir, pr_dir)

    threads = []
    for _ in range(num_procs):
        thread = Thread(target=compare, args=(base_dir, pr_dir, output_dir, files, pr_number, test_number, release_format))
        thread.start()
        threads.append(thread)

    [thread.join() for thread in threads]

    COMPARISON_RESULTS.sort(key=lambda k: float(k['workflow']))

def compare(base_dir, pr_dir, output_dir, files, pr_number, test_number, release_format):
    while files:
        try:
            file_name = files.pop()
            command = ['compareHistograms.py', '-b', os.path.join(base_dir, file_name), \
                '-p', os.path.join(pr_dir, file_name), '-o', output_dir, '-n', pr_number, '-t', test_number, '-r', release_format]
            print('Running comparison:')
            print(' '.join(command))
            
            output = subprocess.check_output(command).decode()

            output_elements = output.split('\n')[1:]
            base_output_filename = output_elements[0]
            pr_output_filename = output_elements[1]
            run_nr = base_output_filename.split('_')[2].lstrip('R').lstrip('0')
            output_numbers = output_elements[2].split(' ')
            
            workflow = os.path.basename(os.path.dirname(os.path.join(base_dir, file_name))).split('_')[0]
            base_dataset = '/' + '/'.join(base_output_filename.rstrip('.root').split('__')[1:])
            pr_dataset = '/' + '/'.join(pr_output_filename.rstrip('.root').split('__')[1:])
            
            cmssw_version = '_'.join(release_format.split('_')[:4])
            cmssw_version = cmssw_version[:-1] + 'x'
            root_file_dir_in_gui = 'ROOT/RelValData/%s/' % cmssw_version
            if 'R000000001__RelVal' in base_output_filename:
                root_file_dir_in_gui = 'ROOT/RelVal/%s/' % cmssw_version

            base_file_path_in_gui = root_file_dir_in_gui + base_output_filename
            pr_file_path_in_gui = root_file_dir_in_gui + pr_output_filename
            
            COMPARISON_RESULTS.append({'workflow': workflow, 'base_dataset': base_dataset, 'pr_dataset': pr_dataset, 'run_nr': run_nr,\
                'changed_elements': int(output_numbers[0]), 'removed_elements': int(output_numbers[1]), 'added_elements': int(output_numbers[2]),
                'base_file_path_in_gui': base_file_path_in_gui, 'pr_file_path_in_gui': pr_file_path_in_gui})
        except Exception as ex:
            print('Exception comparing two root files: %s' % ex)
    
def get_file_pairs(base_dir, pr_dir):
    base_files = glob.glob(os.path.join(base_dir, '*.*_*/DQM_*.root'))
    pr_files = glob.glob(os.path.join(pr_dir, '*.*_*/DQM_*.root'))

    # Remove base directories and leave
    # only parts of paths that are same
    base_files = [ os.path.relpath(x, base_dir) for x in base_files ]
    pr_files =   [ os.path.relpath(x, pr_dir) for x in pr_files ]
    
    # Find intersection
    return [value for value in base_files if value in pr_files]

def upload_to_gui(output_dir, num_procs):
    base_files = glob.glob(os.path.join(output_dir, 'base/*.root'))
    pr_files = glob.glob(os.path.join(output_dir, 'pr/*.root'))

    files = base_files + pr_files

    print('Files to be uploaded:')
    print(files)
    
    for _ in range(min(num_procs, len(files))):
        thread = Thread(target=upload, args=(files,))
        thread.start()
    
def upload(files):
    while files:
        try:
            file = files.pop()
            command = ['visDQMUpload.py', 'https://cmsweb.cern.ch/dqm/dev', file]
            print('Uploading output:')
            print(' '.join(command))
            
            subprocess.call(command)
            print('')
        except Exception as ex:
            # This might throw when another thread pops the last filename immediately after this one
            # started the loop. In this case this exception can be safely ignored.
            print('Exception uploading a file: %s' % ex)

def generate_summary_html(output_dir, pr_list, summary_dir):
    template_file_path = os.path.join(os.getenv('CMSSW_BASE'), 'src', 'DQMServices', 'FileIO', 'scripts', 'dqm-histo-comparison-summary-template.html')
    if not os.path.isfile(template_file_path):
        template_file_path = os.path.join(os.getenv('CMSSW_RELEASE_BASE'), 'src', 'DQMServices', 'FileIO', 'scripts', 'dqm-histo-comparison-summary-template.html')
    template_file = open(template_file_path, 'r')
    result = template_file.read()

    result = result.replace('$PR_LIST$', pr_list)

    table_items = ''
    total_changes = 0
    
    for comp in COMPARISON_RESULTS:
        total_changes += comp['removed_elements'] + comp['added_elements'] + comp['changed_elements']
        baseline_count = comp['changed_elements'] + comp['removed_elements']
        pr_count = comp['changed_elements'] + comp['added_elements']
        overlay_count = baseline_count

        # Make urls
        base_url = 'https://cmsweb.cern.ch/dqm/dev/start?runnr=%s;dataset%%3D%s;sampletype%%3Doffline_relval;workspace%%3DEverything;' % (comp['run_nr'], comp['base_dataset'])
        pr_url = 'https://cmsweb.cern.ch/dqm/dev/start?runnr=%s;dataset%%3D%s;sampletype%%3Doffline_relval;workspace%%3DEverything;' % (comp['run_nr'], comp['pr_dataset'])
        overlay_url = 'https://cmsweb.cern.ch/dqm/dev/start?runnr=%s;dataset%%3D%s;referenceshow%%3Dall;referencenorm=False;referenceobj1%%3Dother::%s::;sampletype%%3Doffline_relval;workspace%%3DEverything;' \
            % (comp['run_nr'], comp['pr_dataset'], comp['base_dataset'])
        base_raw_url = 'https://cmsweb.cern.ch/dqm/dev/jsroot/index.htm?file=https://cmsweb.cern.ch/dqm/dev/data/browse/%s' % comp['base_file_path_in_gui']
        pr_raw_url = 'https://cmsweb.cern.ch/dqm/dev/jsroot/index.htm?file=https://cmsweb.cern.ch/dqm/dev/data/browse/%s' % comp['pr_file_path_in_gui']

        table_items += '        <tr>\n'
        table_items += '            <td><a href="%s" target="_blank">%s baseline GUI</a><span> (%s)</span></td>\n' % (base_url, comp['workflow'], baseline_count)
        table_items += '            <td><a href="%s" target="_blank">%s pr GUI</a><span> (%s)</span></td>\n' % (pr_url, comp['workflow'], pr_count)
        table_items += '            <td><a href="%s" target="_blank">%s overlay GUI</a><span> (%s)</span></td>\n' % (overlay_url, comp['workflow'], overlay_count)
        table_items += '            <td><a href="%s" target="_blank">%s baseline rootjs</a><span> (%s)</span></td>\n' % (base_raw_url, comp['workflow'], baseline_count)
        table_items += '            <td><a href="%s" target="_blank">%s pr rootjs</a><span> (%s)</span></td>\n' % (pr_raw_url, comp['workflow'], pr_count)
        table_items += '            <td><span class="removed">-%s</span><span class="added">+%s</span><span class="changed">%s</span></td>\n' \
            % (comp['removed_elements'], comp['added_elements'], comp['changed_elements'])
        table_items += '        </tr>\n'

    result = result.replace('$TOTAL_CHANGES$', str(total_changes))
    result = result.replace('$NUMBER_OF_WORKFLOWS$', str(len(COMPARISON_RESULTS)))
    result = result.replace('$PER_WORKFLOW_LIST$', table_items)
    template_file.close()

    # Write output
    result_file_path = os.path.join(summary_dir, 'dqm-histo-comparison-summary.html')
    if os.path.dirname(result_file_path):
        if not os.path.exists(os.path.dirname(result_file_path)):
            os.makedirs(os.path.dirname(result_file_path))
    summary_file = open(result_file_path, 'w')
    summary_file.write(result)
    summary_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This tool compares DQM monitor elements within DQM files found in base-dir with the ones found in in pr-dir. "
        "All workflow directories are searched for correctly named DQM root files. "
        "Comparison is done bin by bin and output is written to a root files containing only the changes.")
    parser.add_argument('-b', '--base-dir', help='Baseline IB directory', default='basedata/')
    parser.add_argument('-p', '--pr-dir', help='PR directory', default='prdata/')
    parser.add_argument('-o', '--output-dir', help='Comparison root files output directory', default='dqmHistoComparisonOutput')
    parser.add_argument('-j', '--nprocs', help='Number of processes', default=1, type=int)
    parser.add_argument('-n', '--pr-number', help='This is obsolete and should NOT be used.', required=False)
    parser.add_argument('-t', '--test-number', help='Unique test number to distinguish different comparisons of the same PR.', default='1')
    parser.add_argument('-r', '--release-format', help='Release format in this format: CMSSW_10_5_X_2019-02-17-0000')
    parser.add_argument('-s', '--summary-dir', help='Directory where summary with all links will be saved', default='')
    parser.add_argument('-l', '--pr-list', help='A list of PRs participating in the comparison', default='')
    args = parser.parse_args()

    # Get the number of the PR which triggered the comparison
    pr_number = 'Unknown'
    try:
        pr_number = args.pr_list.split(' ')[0].split('/')[1].replace('#', '_')
    except:
        pass

    release_format = args.release_format
    if not release_format:
        try:
            release_format = os.environ['CMSSW_VERSION']
        except:
            print('You are not in a CMSSW release. Please provide a valid release-format (-r option)')
            os._exit(1)

    collect_and_compare_files(args.base_dir, args.pr_dir, args.output_dir, args.nprocs, pr_number, args.test_number, release_format)
    upload_to_gui(args.output_dir, args.nprocs)
    generate_summary_html(args.output_dir, args.pr_list, args.summary_dir)
