import argparse as _argparse
import textwrap as _textwrap

# argparse's formatters remove newlines from comand descriptions, so we define a new one
class HelpFormatterRespectNewlines(_argparse.HelpFormatter):
    """Help message formatter which retains line breaks in argument descriptions.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _split_lines(self, text, width):
        lines = []
        for line in text.splitlines():
          line = self._whitespace_matcher.sub(' ', line).strip()
          lines.extend( _textwrap.wrap(line, width) )
        return lines

# argparse's formatters are not really able to discover the terminale size, so we override them
def FixedWidthFormatter(formatter, width):
  """Adaptor for argparse formatters using an explicit fixed width
  """
  def f(*args, **keywords):
    # add or replace the "width" parameter
    keywords['width'] = width
    return formatter(*args, **keywords)

  return f

