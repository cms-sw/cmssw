from __future__ import print_function
# word-wrap functions
# written by Mike Brown
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/148061
from builtins import range
import math,re
from functools import reduce
def wrap_always(text, width):
    """
    A simple word-wrap function that wraps text on exactly width characters.
    It doesn't split the text in words.
    """
    return '\n'.join([ text[width*i:width*(i+1)] \
                       for i in range(int(math.ceil(1.*len(text)/width))) ])

def wrap_onspace(text,width):
    """
    A word-wrap function that preserves existing line breaks
    and most spaces in the text. Expects that existing line
    breaks are posix newlines (\n).
    """
    return reduce(lambda line, word, width=width: '%s%s%s' %
                  (line,
                   ' \n'[(len(line[line.rfind('\n')+1:])
                          + len(word.split('\n',1)[0]
                                ) >= width)],
                   word),
                  text.split(' ')
                  )
def wrap_onspace_strict(text, width):
    """
    Similar to wrap_onspace, but enforces the width constraint:
    words longer than width are split.
    """
    wordRegex = re.compile(r'\S{'+str(width)+r',}')
    return wrap_onspace(wordRegex.sub(lambda m: wrap_always(m.group(),width),text),width)

if __name__ == '__main__':
    print(wrap_always('1234567\ntrtyu43222',5))
    print(''.join(['-']*5)+'|')
    print(wrap_onspace('1234567\ntrtyu43222',5))
    print(''.join(['-']*5)+'|')
    print(wrap_onspace_strict('1234567\ntrtyu43222',5))
    print(''.join(['-']*5)+'|')
