#!/usr/bin/env python

import os
import sys
import re
import fcntl
import glob
import shutil
import datetime
import subprocess

if "CMSSW_BASE" not in os.environ:
    print "You need to source the CMSSW environment first."
    sys.exit(1)

from Alignment.MillePedeAlignmentAlgorithm.alignmentsetup.helper \
    import checked_out_MPS

required_version = (2,7)
if sys.version_info < required_version:
    print "Your Python interpreter is too old. Need version 2.7 or higher."
    sys.exit(1)

import argparse


################################################################################
def main(argv = None):
    """Main routine of the script.

    Arguments:
    - `argv`: arguments passed to the main routine
    """

    if argv == None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Setup a new alignment campaign in the MPproduction area.")
    parser.add_argument("-d", "--description", dest="description", required=True,
                        help="comment to describe the purpose of the campaign")
    parser.add_argument("-t", "--data-type", dest="type", required=True,
                        metavar="TYPE", choices=["MC", "data"],
                        help="type of the input data (choices: %(choices)s)")
    parser.add_argument("-c", "--copy", dest="copy", metavar="CAMPAIGN",
                        help="input campaign (optional)")
    args = parser.parse_args(argv)


    if os.path.basename(os.path.normpath(os.getcwd())) != "MPproduction":
        print ">>> Cannot create a campaign outside of the 'MPproduction' area."
        print ">>> Please change to the 'MPproduction' directory first."
        sys.exit(1)

    if len(args.description.strip()) == 0:
        print ">>> Please provide a non-empty description of the campaign"
        sys.exit(1)

    MPS_dir = os.path.join("src", "Alignment", "MillePedeAlignmentAlgorithm")
    args.checked_out = checked_out_MPS()
    if args.checked_out[0]:
        MPS_dir = os.path.join(os.environ["CMSSW_BASE"], MPS_dir)
    else:
        MPS_dir = os.path.join(os.environ["CMSSW_RELEASE_BASE"], MPS_dir)
    args.MPS_dir = MPS_dir

    mp_regex = re.compile(r"mp([0-9]+).*")
    all_campaign_numbers = sorted(map(lambda x: get_first_match(mp_regex, x),
                                      os.listdir(".")))
    next_number = (0
                   if len(all_campaign_numbers) == 0
                   else sorted(all_campaign_numbers)[-1] + 1)

    while True:
        try:
            number_of_digits = len(str(next_number))
            number_of_digits = 4 if number_of_digits <= 4 else number_of_digits
            next_campaign = "mp{{0:0{0}d}}".format(number_of_digits)
            next_campaign = next_campaign.format(next_number)
            os.makedirs(next_campaign)
            print ">>> Created new campaign:", next_campaign

            campaign_list = "MP_ali_list.txt"
            with open(campaign_list, "a") as f:
                add_campaign(f, next_campaign, args)
            backup_dir = ".MP_ali_list"
            try:
                os.makedirs(backup_dir)
            except OSError as e:
                if e.args == (17, 'File exists'):
                    pass
                else:
                    raise
            shutil.copy(campaign_list, backup_dir)
            print "    - updated campaign list '"+campaign_list+"'"

            if args.copy is None:
                copy_default_templates(MPS_dir, next_campaign)
            else:
                copied_files = []
                for ext in ("py", "ini", "txt"):
                    for config_file in glob.glob(args.copy+"/*."+ext):
                        copied_files.append(os.path.basename(config_file))
                        shutil.copy(config_file, next_campaign)
                if len(copied_files) == 0:
                    print "    - no configuration files for '"+args.copy+"'"
                    copy_default_templates(MPS_dir, next_campaign)
                else:
                    print "    - copied configuration files from",
                    print "'"+args.copy+"':", ", ".join(copied_files)

        except OSError as e:
            if e.args == (17, 'File exists'):
                next_number += 1 # someone created a campaign ~at the same time
                continue
            else:
                raise
        break


################################################################################
def get_first_match(regex, directory):
    """
    Checks if `directory` matches `regex` and returns the first match converted
    to an integer. If it does not match -1 is returned.

    Arguments:
    - `regex`: Regular expression to be tested against
    - `directory`: name of the directory under test
    """

    result = regex.search(directory)
    if result is None:
        return -1
    else:
        return int(result.group(1))


def add_campaign(campaign_file, campaign, args):
    """Adds a line with campaign information from `args` to `campaign_file`.

    Arguments:
    - `campaign_file`: output file
    - `campaign`: name of the campaign
    - `args`: command line arguments for this campaign
    """

    campaign_info = campaign.ljust(10)
    campaign_info += os.environ["USER"].ljust(12)
    campaign_info += datetime.date.today().isoformat().ljust(11)

    version = os.environ["CMSSW_VERSION"]
    if args.checked_out[1]:
        local_area = os.path.join(os.environ["CMSSW_BASE"], "src")
        with open(os.devnull, 'w') as devnull:
            # check which tags (-> e.g. release names) point at current commit
            p = subprocess.Popen(["git", "tag", "--points-at", "HEAD"],
                                 cwd = local_area, stdout=subprocess.PIPE,
                                 stderr=devnull)
            tags = p.communicate()[0].split()
            # check for deleted, untracked, modified files respecting .gitignore:
            p = subprocess.Popen(["git", "ls-files", "-d", "-o", "-m",
                                  "--exclude-standard"],
                                 cwd = local_area, stdout=subprocess.PIPE,
                                 stderr=devnull)
            files = p.communicate()[0].split()
            # check for staged tracked files:
            p = subprocess.Popen(["git", "diff", "--name-only", "--staged"],
                                 cwd = local_area, stdout=subprocess.PIPE,
                                 stderr=devnull)
            files.extend(p.communicate()[0].split())
        if version not in tags or len(files) != 0:
            version += " (mod.)"

    campaign_info += version.ljust(34)
    campaign_info += args.type.ljust(17)
    campaign_info += args.description.strip() + "\n"

    fcntl.flock(campaign_file, fcntl.LOCK_EX)
    campaign_file.write(campaign_info)
    fcntl.flock(campaign_file, fcntl.LOCK_UN)


def copy_default_templates(MPS_dir, next_campaign):
    """Copies the default configuration templates.

    Arguments:
    - `MPS_dir`: location of the default templates
    - `next_campaign`: destination for the copy operation
    """

    default_conf_dir = os.path.join(MPS_dir, "test")
    for f in ("universalConfigTemplate.py", "alignment_config.ini"):
        shutil.copy(os.path.join(default_conf_dir, f), next_campaign)
    print "    - copied default configuration templates from",
    print "'"+default_conf_dir+"'"


################################################################################
if __name__ == "__main__":
    main()
