import os
from TkAlExceptions import AllInOneError

####################--- Helpers ---############################
#replaces .oO[id]Oo. by map[id] in target
def replaceByMap(target, map):
    result = target
    for id in map:
        #print "  "+id+": "+map[id]
        lifeSaver = 10e3
        iteration = 0
        while ".oO[" in result and "]Oo." in result:
            for id in map:
                result = result.replace(".oO["+id+"]Oo.",map[id])
                iteration += 1
            if iteration > lifeSaver:
                problematicLines = ""
                print map.keys()
                for line in result.splitlines():
                    if  ".oO[" in result and "]Oo." in line:
                        problematicLines += "%s\n"%line
                raise AllInOneError, "Oh Dear, there seems to be an endless loop in replaceByMap!!\n%s\nrepMap"%problematicLines
    return result

#excute [command] and return output
def getCommandOutput2(command):
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        raise RuntimeError, '%s failed w/ exit code %d' % (command, err)
    return data

#check if a directory exsits on castor
def castorDirExists(path):
    if path[-1] == "/":
        path = path[:-1]
    containingPath = os.path.join( *path.split("/")[:-1] )
    dirInQuestion = path.split("/")[-1]
    try:
        rawLines =getCommandOutput2("rfdir /"+containingPath).splitlines()
    except RuntimeError:
        return False
    for line in rawLines:
        if line.split()[0][0] == "d":
            if line.split()[8] == dirInQuestion:
                return True
    return False
