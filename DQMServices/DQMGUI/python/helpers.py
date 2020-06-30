import os
import time
import logging
import contextvars
from collections import namedtuple
from inspect import getframeinfo, stack
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool


class Timed():
    """A helper that will measure wall clock time between enter and exit methods."""

    def __init__(self):
        caller = getframeinfo(stack()[1][0])
        self.filename = os.path.basename(caller.filename)
        self.lineno = caller.lineno

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        elapsed = elapsed * 1000
        print('%s:%s - %s ms' % (self.filename, self.lineno, elapsed))

# `logged` uses some "task local" variables to track a global request id and
# keep track of how calls are nested.
logged_curid = [0]
logged_reqid = contextvars.ContextVar("logged_reqid", default=None)
logged_depth = contextvars.ContextVar("logged_depth", default=0)


def logged(fn):
    """ A decorator to write timing information to a log. """
    logger = logging.getLogger("helpers.logged")
    async def wrapped(*posargs, **kwargs):
        showargs = [repr(arg) for arg in posargs if isinstance(arg, str) or isinstance(arg, int) or isinstance(arg, tuple)]
        showargs += [repr(arg) for arg in kwargs.values() if isinstance(arg, str) or isinstance(arg, int) or isinstance(arg, tuple)]

        reqid = logged_reqid.get()
        if not reqid:
            logged_curid[0] += 1
            reqid = logged_curid[0]
            logged_reqid.set(reqid)
        depth = logged_depth.get() + 1
        logged_depth.set(depth)

        msg = f"{reqid}{' ' * depth}{fn.__qualname__}({', '.join(showargs)})"
        start_time = time.time()
        ok = "FAIL"
        try:
            ret = await fn(*posargs, **kwargs)
            ok = "OK"
        finally:
            elapsed = time.time() - start_time
            logged_depth.set(depth - 1)
            logger.info(f"{msg} [{ok} {elapsed*1000:.1f}ms]")
        return ret
    return wrapped


def get_api_error(message):
    """Returns an object that is returned by the API to signify an error."""
    return { 'message': message }


def get_base_release_dir():
    """
    Returns an absolute path to a CMSSW directory.
    If DQMServices/DQMGUI package is checked out, returns a path to user checkout.
    If DQMServices/DQMGUI package is not checked out, return a path to base CMSSW release.
    """

    user_path = os.environ.get('CMSSW_BASE')
    if user_path:
        if os.path.isdir(os.path.join(user_path, 'src/DQMServices/DQMGUI')):
            return user_path
    return os.environ.get('CMSSW_RELEASE_BASE')


def get_absolute_path(to=''):
    """
    Returns an absolute path to a specified directory or file. 
    Specified path must be a relative path starting at DQMServices/DQMGUI/python/
    """

    base = get_base_release_dir()
    directory = 'src/DQMServices/DQMGUI/python/'
    if base:
        return os.path.join(base, directory, to)
    return os.path.join(directory, to)


def binary_search(array, target, key=None):
    """
    Binary search implementation. Returns an index of an element if found, otherwise -1.
    If key is passed, value will be taken by that key from every item in the array.
    If decode is True, every item in the array is utf-8 decoded.
    """

    first = 0
    last = len(array) - 1

    while first <= last:
        mid = (first + last)//2

        if key:
            current = array[mid][key]
        else:
            current = array[mid]

        if current == target:
            return mid
        else:
            if target < current:
                last = mid - 1
            else:
                first = mid + 1
    return -1


def binary_search_qtests(me_names, me_path):
    """
    Binary search implementation. me_names array has to be an array of binary
    strings containing ME names.
    Returns a tuple of indices pointing to the QTest MEInfo items of a given me_path.
    """

    first = 0
    last = len(me_names) - 1
    index = -1
    target = me_path + b'\0.'

    while first <= last:
        mid = (first + last)//2

        if me_names[mid][:len(target)] == target:
            index = mid
            break
        else:
            if target < me_names[mid][:len(target)]:
                last = mid - 1
            else:
                first = mid + 1

    if index == -1:
        return tuple()

    # Found an index, now linearly search nearby items to get all values
    result = (index,)

    potential_index = 1
    while index + potential_index < len(me_names):
        if me_names[index + potential_index][:len(target)] == target:
            result += (index + potential_index,)
        else:
            break

    potential_index = 1
    while index - potential_index >= 0:
        if me_names[index - potential_index][:len(target)] == target:
            result += (index - potential_index,)
        else:
            break

    return result


def parse_run_lumi(runlumi):
    """
    Run/lumi pair is passed to the API in this format: run:lumi
    This method parses such string and returns a tuple (run, lumi).
    If lumi is not passed, it's assumed that it's value is 0.
    """
    
    if not runlumi:
        return None, None
    elif ':' in runlumi:
        parts = runlumi.split(':')
        if not parts[1]:
            return parts[0], 0
        else:
            return parts[0], parts[1]
    else:
        return runlumi, 0


