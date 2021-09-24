"""
File that contains utility functions used by various modules, but that do not fit into any single module.
"""
import datetime

def to_timestamp(obj):
    """
    Takes a datetime object and outputs a timestamp string with the format Y-m-d H:m:S.f
    """
    return obj.strftime('%d-%b-%y %I:%M:%S.%f %p') if isinstance(obj, datetime.datetime) else obj

def to_datetime(date_string):
    """
    Takes a date string with the format Y-m-d H:m:S.f and gives back a datetime.datetime object
    """
    return datetime.datetime.strptime(date_string.replace(",", "."), "%d-%b-%y %I:%M:%S.%f %p")

def friendly_since(time_type, since):
    """
    Takes a since and, if it is Run-based expressed as Lumi-based,
    returns the run number.
    Otherwise, returns the since without transformations.
    """
    if time_type == "Run" and (since & 0xffffff) == 0:
        return since >> 32
    else:
        return since