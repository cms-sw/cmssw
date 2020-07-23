"""
An implementation of async lru cache that generates a cache miss if cached value is older than specified threshold.

alru_cache_timed decorator will inspect kwargs and if notOlderThan key is there, its value will be used as a timestamp.
If cached value exists and it was cached after notOlderThan, it's a cache hit. 
Otherwise it's a cache miss and cached value will get invalidated.
All timestamps must be provided in UTC time zone!!!
If no notOlderThan value is found or notOlderThan is None, utcnow() - DEFAULT_NOT_OLDER_THAN_SECONDS_DELTA will be used.
notOlderThan will not be taken into account when creating a key for cache dict.
"""

import asyncio
from datetime import datetime
from collections import OrderedDict
from functools import _CacheInfo, _make_key, partial, wraps

__all__ = ('alru_cache_timed',)

DEFAULT_NOT_OLDER_THAN_SECONDS_DELTA = 3600 # 1 hour


def unpartial(fn):
    while hasattr(fn, 'func'):
        fn = fn.func
    return fn


def _cache_invalidate(wrapped, typed, *args, **kwargs):
    key = _make_key(args, kwargs, typed)

    exists = key in wrapped._cache

    if exists:
        wrapped._cache.pop(key)

    return exists


def _cache_clear(wrapped):
    wrapped.hits = wrapped.misses = 0
    wrapped._cache = OrderedDict()


def _cache_info(wrapped, maxsize):
    return _CacheInfo(
        wrapped.hits,
        wrapped.misses,
        maxsize,
        len(wrapped._cache),
    )


def __cache_touch(wrapped, key):
    try:
        wrapped._cache.move_to_end(key)
    except KeyError:  # not sure is it possible
        pass


def _cache_hit(wrapped, key):
    wrapped.hits += 1
    __cache_touch(wrapped, key)


def _cache_miss(wrapped, key):
    wrapped.misses += 1
    __cache_touch(wrapped, key)


def alru_cache_timed(
    fn=None,
    maxsize=128,
    typed=False,
    *,
    cache_exceptions=True,
):
    def wrapper(fn):
        _origin = unpartial(fn)

        if not asyncio.iscoroutinefunction(_origin):
            raise RuntimeError(
                'Coroutine function is required, got {}'.format(fn))

        # functools.partialmethod support
        if hasattr(fn, '_make_unbound_method'):
            fn = fn._make_unbound_method()

        @wraps(fn)
        async def wrapped(*fn_args, **fn_kwargs):
            # If notOlderThan is not provided or None, use default timestamp delta:
            # utcnow() - DEFAULT_NOT_OLDER_THAN_SECONDS_DELTA
            notOlderThan = fn_kwargs.get('notOlderThan', None)
            notOlderThan = int(notOlderThan) if notOlderThan else None
            if not notOlderThan:
                notOlderThan = int(datetime.utcnow().timestamp()) - DEFAULT_NOT_OLDER_THAN_SECONDS_DELTA

            key = _make_key(fn_args, fn_kwargs, typed)

            # We store (mutable) lists in the cache, so that even if an item
            # is evicted before its task finished, other waiting tasks can get
            # the result. We need to keep a reference to the list here, rather
            # than unpacking.
            event_result_ex_time = wrapped._cache.get(key)

            # Check if value was cached after notOlderThan. If it was, it's a hit, 
            # otherwise invalidate and call the function to get a new value
            if event_result_ex_time != None and notOlderThan != None and event_result_ex_time[3] < notOlderThan:
                _cache_invalidate(wrapped, typed, *fn_args, **fn_kwargs)
                event_result_ex_time = None

            if event_result_ex_time is None:
                # List items: 
                # Future to await, already awaited result, exception, UTC timestamp when cached
                event_result_ex_time = [asyncio.Event(), None, None, int(datetime.utcnow().timestamp())]
                # logically there is a possible race between get above and 
                # insert here. Make sure there is no `await` in between.
                wrapped._cache[key] = event_result_ex_time
                _cache_miss(wrapped, key)

                if maxsize is not None and len(wrapped._cache) > maxsize:
                    wrapped._cache.popitem(last=False)

                try:
                    res = await fn(*fn_args, **fn_kwargs)
                    event_result_ex_time[1] = res
                    return res
                except Exception as e:
                    event_result_ex_time[2] = e
                    # Even with cache_exceptions=False, we don't retry for
                    # requests that happened *in parallel* to the first one.
                    # Only once the initial taks has failed, the next request
                    # will not hit the cache.
                    if cache_exceptions == False:
                        _cache_invalidate(wrapped, typed, *fn_args, **fn_kwargs)
                    # make sure to pass on the exception to get a proper traceback.
                    raise
                finally:
                    # now at least result or exc is set, and we can release others.
                    event_result_ex_time[0].set()
            else:
                _cache_hit(wrapped, key)
                # this will return immediately if the task is done, no need to
                # check manually.
                await event_result_ex_time[0].wait()
                if event_result_ex_time[2]:
                    raise event_result_ex_time[2]
                else:
                    return event_result_ex_time[1]

        _cache_clear(wrapped)
        wrapped._origin = _origin
        wrapped.cache_info = partial(_cache_info, wrapped, maxsize)
        wrapped.cache_clear = partial(_cache_clear, wrapped)
        wrapped.invalidate = partial(_cache_invalidate, wrapped, typed)

        return wrapped

    if fn is None:
        return wrapper

    if callable(fn) or hasattr(fn, '_make_unbound_method'):
        return wrapper(fn)

    raise NotImplementedError('{} decorating is not supported'.format(fn))
