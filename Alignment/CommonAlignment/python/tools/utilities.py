from __future__ import print_function

def cache(function):
    cache = {}
    def newfunction(*args, **kwargs):
        try:
            return cache[args, tuple(sorted(kwargs.iteritems()))]
        except TypeError:
            print(args, tuple(sorted(kwargs.iteritems())))
            raise
        except KeyError:
            cache[args, tuple(sorted(kwargs.iteritems()))] = function(*args, **kwargs)
            return newfunction(*args, **kwargs)
    newfunction.__name__ = function.__name__
    return newfunction
