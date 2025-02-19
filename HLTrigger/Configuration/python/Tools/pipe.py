import shlex as _shlex
import subprocess as _subprocess


def pipe(cmdline, input = None):
  """
  wrapper around subprocess to simplify te interface
  """
  args = _shlex.split(cmdline)
  if input is not None:
    command = _subprocess.Popen(args, stdin = _subprocess.PIPE, stdout = _subprocess.PIPE, stderr = None)
  else:
    command = _subprocess.Popen(args, stdin = None, stdout = _subprocess.PIPE, stderr = None)
  (out, err) = command.communicate(input)
  return out
