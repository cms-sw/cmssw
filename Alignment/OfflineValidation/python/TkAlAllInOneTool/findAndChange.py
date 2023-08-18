import os
from FWCore.ParameterSet.pfnInPath import pfnInPath
import ROOT

##############################################
def digest_path(path):
##############################################
    """ Ensure that everything is done for path to exist
    Arguments:
    - path: String that can be both directory and file 
    Return:
    - general: environmental variables are expanded
    - directory: is checked to exist
    - file: is checked to exist with backup directory being searched in cms-data
    """
    # sanity check for string argument
    if not isinstance(path, str):
        return path

    path_expanded = os.path.expandvars(path)

    # split path in folders
    protocol = ""
    if "://" in path_expanded:
        protocol = path_expanded.split("://")[0]+"://"
        path_d = path_expanded.split("://")[1]
    elif ":" in path_expanded:
        protocol = path_expanded.split(":")[0]+':'
        path_d = ":".join(path_expanded.split(":")[1:])
        # Similar to just `split(':')[1]`, but handles the case in which the rest of the path contains one or more ':'
    else:
        path_d = path_expanded

    path_s = path_d.split(os.sep)

    placeholderIdx = []
    for ipart,part in enumerate(path_s):
        # Look for {} placeholder to be replaced internally 
        if "{}" in part:
            placeholderIdx.append(ipart)

    # re-join folders into full path 
    # only check path up to first placeholder occurence
    if len(placeholderIdx) > 0:
        path_to_check = os.path.join(*path_s[:placeholderIdx[0]])
        # re add front / if needed
        if path_d.startswith(os.sep):
            path_to_check = os.sep + path_to_check
    else:
        path_to_check = path_d

    # check for path to exist
    if not os.path.exists(path_to_check) and "." in os.path.splitext(path_to_check)[-1]:
        # in case of directory pointing to file try backup
        _file = pfnInPath(path_to_check)
        if "file:" in _file:
            return _file.split(":")[-1]

    # re add protocol declaration
    if protocol != "": path_d = protocol + path_d

    # if all is OK return path to directory or file
    return path_d

#########################################
def get_root_color(value):
#########################################
    """
       Returns an integer correspondig to the ROOT color
    """
    if(isinstance(value, str)):
        if(value.isdigit()):
            return int(value)
        elif('-' in value):
            pre, op, post = value.partition('-')
            return get_root_color(pre) - get_root_color(post)
        elif('+' in value):
            pre, op, post = value.partition('+')
            return get_root_color(pre) + get_root_color(post)
        else:
            return getattr(ROOT.EColor, value)
    else:
        return int(value)

#########################################
def get_all_keys(var):
#########################################
    """
       Generate all keys for nested dictionary
       - reserved keywords are not picked up
    """
    reserved_keys = ["customrighttitle","title"]
    if hasattr(var,'items'):
        for k, v in var.items():
            if k in reserved_keys: continue
            if isinstance(v, dict):
                for result in get_all_keys(v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in get_all_keys(d):
                        yield result
            else:
                yield k

####################################################
def find_and_change(keys, var, alt=digest_path):
####################################################
    """Perform effective search for keys in nested dictionary
       - if key is found, corresponding value is "digested"
       - generator is returned for printout purpose only
       - original var is overwritten
    """
    if hasattr(var,'items'):
        if len(keys) == 0:
            for key in get_all_keys(var):
                keys.append(key)
        for key in keys:
            for k, v in var.items():
                if k == key:
                    if "color" in key:
                        if isinstance(v,list):
                            var[k] = [get_root_color(_v) for _v in v]
                        else:
                            var[k] = get_root_color(v)
                    else:    
                        if isinstance(v,list):
                            var[k] = [alt(_v) for _v in v]
                        else:
                            var[k] = alt(v)
                    yield alt(v)
                if isinstance(v, dict):
                    for result in find_and_change([key], v):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in find_and_change([key], d):
                            yield result
