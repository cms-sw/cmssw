import os
import re
import sys
import shutil
import importlib
import sqlalchemy
import subprocess
import CondCore.Utilities.conddblib as conddb
from functools import reduce


def create_single_iov_db(inputs, run_number, output_db):
    """Create an sqlite file with single-IOV tags for alignment payloads.

    Arguments:
    - `inputs`: dictionary with input needed for payload extraction
    - `run_number`: run for which the IOVs are selected
    - `output_db`: name of the output sqlite file
    """

    # find the IOV containing `run_number`
    for record,tag in inputs.iteritems():
        run_is_covered = False
        for iov in reversed(tag["iovs"]):
            if iov <= run_number:
                tag["since"] = str(iov)
                run_is_covered = True
                break
        if not run_is_covered:
            msg = ("Run number {0:d} is not covered in '{1:s}' ({2:s}) from"
                   " '{3:s}'.".format(run_number, tag["tag"], record,
                                      global_tag))
            print msg
            print "Aborting..."
            sys.exit(1)

    result = {}
    remove_existing_object(output_db)

    for record,tag in inputs.iteritems():
        result[record] = {"connect": "sqlite_file:"+output_db,
                          "tag": "_".join([tag["tag"], tag["since"]])}

        if tag["connect"] == "pro":
            source_connect = "frontier://FrontierProd/CMS_CONDITIONS"
        elif tag["connect"] == "dev":
            source_connect = "frontier://FrontierPrep/CMS_CONDITIONS"
        else:
            source_connect = tag["connect"]

        cmd = ("conddb_import",
               "-f", source_connect,
               "-c", result[record]["connect"],
               "-i", tag["tag"],
               "-t", result[record]["tag"],
               "-b", str(run_number),
               "-e", str(run_number))
        run_checked(cmd)
    if len(inputs) > 0:
        run_checked(["sqlite3", output_db, "update iov set since=1"])

    return result


def run_checked(cmd, suppress_stderr = False):
    """Run `cmd` and exit in case of failures.

    Arguments:
    - `cmd`: list containing the strings of the command
    - `suppress_stderr`: suppress output from stderr
    """

    try:
        with open(os.devnull, "w") as devnull:
            if suppress_stderr:
                subprocess.check_call(cmd, stdout = devnull, stderr = devnull)
            else:
                subprocess.check_call(cmd, stdout = devnull)
    except subprocess.CalledProcessError as e:
        print "Problem in running the following command:"
        print " ".join(e.cmd)
        sys.exit(1)


def get_process_object(cfg):
    """Returns cms.Process object defined in `cfg`.

    Arguments:
    - `cfg`: path to CMSSW config file
    """

    sys.path.append(os.path.dirname(cfg)) # add location to python path
    cache_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")    # suppress unwanted output
    try:
        __configuration = \
            importlib.import_module(os.path.splitext(os.path.basename(cfg))[0])
    except Exception as e:
        print "Problem detected in configuration file '{0}'.".format(cfg)
        raise e
    sys.stdout = cache_stdout
    sys.path.pop()                        # clean up python path again
    try:
        os.remove(cfg+"c")                # try to remove temporary .pyc file
    except OSError as e:
        if e.args == (2, "No such file or directory"): pass
        else: raise

    return __configuration.process


def make_unique_runranges(ali_producer):
    """Derive unique run ranges from AlignmentProducer PSet.

    Arguments:
    - `ali_producer`: cms.PSet containing AlignmentProducer configuration
    """

    if (hasattr(ali_producer, "RunRangeSelection") and
        len(ali_producer.RunRangeSelection) > 0):
        iovs = set([int(iov)
                    for sel in ali_producer.RunRangeSelection
                    for iov in sel.RunRanges])
        if len(iovs) == 0: return [1] # single IOV starting from run 1
        return sorted(iovs)
    else:
        return [1]                    # single IOV starting from run 1


def get_tags(global_tag, records):
    """Get tags for `records` contained in `global_tag`.

    Arguments:
    - `global_tag`: global tag of interest
    - `records`: database records of interest
    """

    if len(records) == 0: return {} # avoid useless DB query

    # check for auto GT
    if global_tag.startswith("auto:"):
        import Configuration.AlCa.autoCond as AC
        try:
            global_tag = AC.autoCond[global_tag.split("auto:")[-1]]
        except KeyError:
            print "Unsupported auto GT:", global_tag
            sys.exit(1)

    # setting up the DB session
    con = conddb.connect(url = conddb.make_url())
    session = con.session()
    GlobalTagMap = session.get_dbtype(conddb.GlobalTagMap)

    # query tag names for records of interest contained in `global_tag`
    tags = session.query(GlobalTagMap.record, GlobalTagMap.tag_name).\
           filter(GlobalTagMap.global_tag_name == global_tag,
                  GlobalTagMap.record.in_(records)).all()

    # closing the DB session
    session.close()

    return {item[0]: {"tag": item[1], "connect": "pro"} for item in tags}


def get_iovs(db, tag):
    """Retrieve the list of IOVs from `db` for `tag`.

    Arguments:
    - `db`: database connection string
    - `tag`: tag of database record
    """

    db = db.replace("sqlite_file:", "").replace("sqlite:", "")
    db = db.replace("frontier://FrontierProd/CMS_CONDITIONS", "pro")
    db = db.replace("frontier://FrontierPrep/CMS_CONDITIONS", "dev")

    con = conddb.connect(url = conddb.make_url(db))
    session = con.session()
    IOV = session.get_dbtype(conddb.IOV)

    iovs = set(session.query(IOV.since).filter(IOV.tag_name == tag).all())
    if len(iovs) == 0:
        print "No IOVs found for tag '"+tag+"' in database '"+db+"'."
        sys.exit(1)

    session.close()

    return sorted([int(item[0]) for item in iovs])


def replace_factors(product_string, name, value):
    """Takes a `product_string` and replaces all factors with `name` by `value`.

    Arguments:
    - `product_string`: input string containing a product
    - `name`: name of the factor
    - `value`: value of the factor
    """

    value = str(value)                                 # ensure it's a string
    return re.sub(r"^"+name+r"$", value,               # single factor
                  re.sub(r"[*]"+name+r"$", r"*"+value, # rhs
                         re.sub(r"^"+name+r"[*]", value+r"*", # lhs
                                re.sub(r"[*]"+name+r"[*]", r"*"+value+r"*",
                                       product_string))))

def compute_product_string(product_string):
    """Takes `product_string` and returns the product of the factors as string.

    Arguments:
    - `product_string`: string containing product ('<factor>*<factor>*...')
    """

    factors = [float(f) for f in product_string.split("*")]
    return str(reduce(lambda x,y: x*y, factors))


def check_proxy():
    """Check if GRID proxy has been initialized."""

    try:
        with open(os.devnull, "w") as dump:
            subprocess.check_call(["voms-proxy-info", "--exists"],
                                  stdout = dump, stderr = dump)
    except subprocess.CalledProcessError:
        return False
    return True


def remove_existing_object(path):
    """
    Tries to remove file or directory located at `path`. If the user
    has no delete permissions, the object is moved to a backup
    file. If this fails it tries 5 times in total and then asks to
    perform a cleanup by a user with delete permissions.

    Arguments:
    - `name`: name of the object to be (re)moved
    """

    if os.path.exists(path):
        remove_method = shutil.rmtree if os.path.isdir(path) else os.remove
        move_method = shutil.move if os.path.isdir(path) else os.rename
        try:
            remove_method(path)
        except OSError as e:
            if e.args != (13, "Permission denied"): raise
            backup_path = path.rstrip("/")+"~"
            for _ in xrange(5):
                try:
                    if os.path.exists(backup_path): remove_method(backup_path)
                    move_method(path, backup_path)
                    break
                except OSError as e:
                    if e.args != (13, "Permission denied"): raise
                    backup_path += "~"
        if os.path.exists(path):
            msg = ("Cannot remove '{}' due to missing 'delete' ".format(path)
                   +"permissions and the limit of 5 backups is reached. Please "
                   "ask a user with 'delete' permissions to clean up.")
            print msg
            sys.exit(1)
