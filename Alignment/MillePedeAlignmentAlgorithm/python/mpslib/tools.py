import os
import sys
import subprocess
import CondCore.Utilities.CondDBFW.shell as shell


def create_single_iov_db(global_tag, run_number, output_db):
    """Create an sqlite file with single-IOV tags for alignment payloads.

    Arguments:
    - `global_tag`: global tag from which to extract the payloads
    - `run_number`: run for which the IOVs are selected
    - `output_db`: name of the output sqlite file
    """

    con = shell.connect()
    tags = con.global_tag_map(global_tag_name = global_tag,
                              record = ["TrackerAlignmentRcd",
                                        "TrackerSurfaceDeformationRcd",
                                        "TrackerAlignmentErrorExtendedRcd"])
    con.close_session()

    tags = {item["record"]: {"name": item["tag_name"]}
            for item in tags.as_dicts()}

    for record,tag in tags.iteritems():
        iovs = con.tag(name = tag["name"]).iovs().as_dicts()
        run_is_covered = False
        for iov in reversed(iovs):
            if iov["since"] <= run_number:
                tag["since"] = str(iov["since"])
                run_is_covered = True
                break
        if not run_is_covered:
            msg = ("Run number {0:d} is not covered in '{1:s}' ({2:s}) from"
                   " '{3:s}'.".format(run_number, tag["name"], record,
                                      global_tag))
            print msg
            print "Aborting..."
            sys.exit(1)

    result = {}
    if os.path.exists(output_db): os.remove(output_db)
    for record,tag in tags.iteritems():
        result[record] = {"connect": "sqlite_file:"+output_db,
                          "tag": "_".join([tag["name"], tag["since"]])}
        cmd = ("conddb_import",
               "-f", "frontier://PromptProd/cms_conditions",
               "-c", result[record]["connect"],
               "-i", tag["name"],
               "-t", result[record]["tag"],
               "-b", str(run_number),
               "-e", str(run_number))
        run_checked(cmd)
    run_checked(["sqlite3", output_db, "update iov set since=1"])

    return result


def run_checked(cmd):
    """Run `cmd` and exit in case of failures.

    Arguments:
    - `cmd`: list containing the strings of the command
    """

    try:
        with open(os.devnull, "w") as devnull:
            subprocess.check_call(cmd, stdout = devnull)
    except subprocess.CalledProcessError as e:
        print "Problem in running the following command:"
        print " ".join(e.cmd)
        sys.exit(1)
