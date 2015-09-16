import signal, time

class TimedOutExc(Exception):
    def __init__(self, value = "Timed Out"):
        self.value = value
    def __str__(self):
        return repr(self.value)

def TimedOutFn(f, timeout, *args, **kwargs):
    def handler(signum, frame):
        raise TimedOutExc()
    
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = f(*args, **kwargs)
    finally:
        signal.signal(signal.SIGALRM, old)
    signal.alarm(0)
    return result


def timed_out(timeout):
    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()
        
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

            time_up = True
            try:
                result = f(*args, **kwargs)
                time_up = False
            finally:
                signal.signal(signal.SIGALRM, old)
                signal.alarm(0)
                if time_up:
                    raise TimedOutExc()
            return result
        
        new_f.func_name = f.func_name
        return new_f

    return decorate
