import argparse
import subprocess
import signal, fcntl
import sys, os, time

# dup and close stdin/stdout
stdin_  = os.dup(sys.stdin.fileno())
stdout_ = os.dup(sys.stdout.fileno())
sys.stdin.close()
sys.stdout.close()

class Timeout(IOError):
    pass

def alarm_handler(signum, frame):
    if signum != 14: return
    raise Timeout("Timeout reached.")

signal.signal(signal.SIGALRM, alarm_handler)

def log(s):
    sys.stderr.write("watchdog: " + s + "\n");
    sys.stderr.flush()

def launch(args):
    fd, wd = os.pipe()

    def preexec():
        os.close(fd)

        env = os.environ
        env["WATCHDOG_FD"] = str(wd)

    p = subprocess.Popen(args.pargs, preexec_fn=preexec, stdin=stdin_, stdout=stdout_)
    os.close(wd)

    while True:
        try:
            signal.alarm(args.t)
            ch = os.read(fd, 1024)
            signal.alarm(0)
            
            if not ch:
                os.close(fd)
                return False, p.wait() # normal exit
            
            log("Received: %s, timer reset." % repr(ch))

        except Timeout, t:
            signal.alarm(0)

            log("Timeout reached, taking action.")

            if p.poll() is None:
                p.send_signal(args.action)

            os.close(fd)
            return True, p.wait()

    for p in open_procs_:
        if p.poll() is None:
            p.send_signal(sig)

def main(args):
    while True:
        killed, ret = launch(args)
        log("Program exitted, killed: %s, code: %d." % (killed, ret, ))

        if killed and args.restart:
            log("Restarting.")
            continue
    
        break


parser = argparse.ArgumentParser(description="Kill/restart the child process if it doesn't out the required string.")
parser.add_argument("-t", type=int, default="2", help="Timeout in seconds.")
parser.add_argument("-s", type=int, default="2000", help="Signal to send.")
parser.add_argument("-r", "--restart",  action="store_true", default=False, help="Restart the process after killing it.")
parser.add_argument("pargs", nargs=argparse.REMAINDER)

group = parser.add_mutually_exclusive_group()
group.add_argument('--term', action='store_const', dest="action", const=signal.SIGTERM, default=signal.SIGTERM)
group.add_argument('--kill', action='store_const', dest="action", const=signal.SIGKILL) 

if __name__ == "__main__":
    args = parser.parse_args()
    #log("Args: %s." % str(args))
    main(args)
