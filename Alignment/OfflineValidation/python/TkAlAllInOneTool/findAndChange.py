import os
from FWCore.ParameterSet.pfnInPath import pfnInPath

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

    # split path in folders
    protocol = ""
    if "://" in path:
        protocol = path.split("://")[0]+"://"
        path_s = path.split("://")[1].split(os.sep)
    else:
        path_s = path.split(os.sep)    

    path_d_s = []
    placeholderIdx = []
    for ipart,part in enumerate(path_s):
        # Look for environment variables such as $CMSSW_BASE
        if part.startswith('$'):
            env_var = part[1:].replace('{', '').replace('}', '')
            path_d_s.append(os.environ[env_var])
        # Look for {} placeholder to be replaced internally 
        elif "{}" in part:
            placeholderIdx.append(ipart)
            path_d_s.append(part)
        else:
            path_d_s.append(part)

    # re-join folders into full path 
    # only check path up to first placeholder occurence
    path_d = os.path.join(*path_d_s)
    if len(placeholderIdx) > 0:
        path_to_check = os.path.join(*path_d_s[:placeholderIdx[0]])
    else:
        path_to_check = path_d

    # re add front / if needed
    if path.startswith(os.sep):
        path_d = os.sep + path_d
        path_to_check = os.sep + path_to_check

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
def get_all_keys(var):
#########################################
    """
       Generate all keys for nested dictionary
    """
    if hasattr(var,'items'):
        for k, v in var.items():
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
