import os
from TkAlExceptions import AllInOneError

####################--- Helpers ---############################
def replaceByMap(target, the_map):
    """This function replaces `.oO[key]Oo.` by `the_map[key]` in target.

    Arguments:
    - `target`: String which contains symbolic tags of the form `.oO[key]Oo.`
    - `the_map`: Dictionary which has to contain the `key`s in `target` as keys
    """

    result = target
    for key in the_map:
        lifeSaver = 10e3
        iteration = 0
        while ".oO[" in result and "]Oo." in result:
            for key in the_map:
                result = result.replace(".oO["+key+"]Oo.",the_map[key])
                iteration += 1
            if iteration > lifeSaver:
                problematicLines = ""
                for line in result.splitlines():
                    if  ".oO[" in result and "]Oo." in line:
                        problematicLines += "%s\n"%line
                msg = ("Oh Dear, there seems to be an endless loop in "
                       "replaceByMap!!\n%s\nrepMap"%problematicLines)
                raise AllInOneError(msg)
    return result


def getCommandOutput2(command):
    """This function executes `command` and returns it output.

    Arguments:
    - `command`: Shell command to be invoked by this function.
    """

    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError, '%s failed w/ exit code %d' % (command, err)
    return data


def castorDirExists(path):
    """This function checks if the directory given by `path` exists.

    Arguments:
    - `path`: Path to castor directory
    """

    if path[-1] == "/":
        path = path[:-1]
    containingPath = os.path.join( *path.split("/")[:-1] )
    dirInQuestion = path.split("/")[-1]
    try:
        rawLines = getCommandOutput2("rfdir /"+containingPath).splitlines()
    except RuntimeError:
        return False
    for line in rawLines:
        if line.split()[0][0] == "d":
            if line.split()[8] == dirInQuestion:
                return True
    return False
