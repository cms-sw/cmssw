import os
import re
import pprint as pprint # for testing

def loadListFromFile (filename):
    """Loads a list of strings from file.  Will append to given list
    if asked."""
    retval = []
    filename = os.path.expanduser (filename)
    if not os.path.exists (filename):
        print "Error: '%s' file does not exist."
        raise RuntimeError, "Bad filename"
    source = open (filename, 'r')        
    for line in source.readlines():
        line = re.sub (r'#.+$', '', line) # remove comment characters
        line = line.strip()
        if len (line):
            retval.append (line)
    source.close()
    return retval


def sectionNofTotal (inputList, currentSection, numSections):
    """Returns the appropriate sublist given the current section
    (1..numSections)"""
    currentSection -= 1 # internally, we want 0..N-1, not 1..N
    size       = len (inputList)
    perSection = size // numSections
    extra      = size %  numSections
    start      = perSection * currentSection
    num        = perSection
    if currentSection < extra:
        # the early sections get an extra item
        start += currentSection
        num   += 1
    else:
        start += extra
    stop = start + num
    return inputList[ start:stop ]


##############################################################################
## ######################################################################## ##
## ##                                                                    ## ##
## ######################################################################## ##
##############################################################################

    
if __name__ == "__main__":
    #############################################
    ## Load and save command line history when ##
    ## running interactively.                  ##
    #############################################
    import os, readline
    import atexit
    historyPath = os.path.expanduser("~/.pyhistory")


    def save_history(historyPath=historyPath):
        import readline
        readline.write_history_file(historyPath)
        if os.path.exists(historyPath):
            readline.read_history_file(historyPath)


    atexit.register(save_history)
    readline.parse_and_bind("set show-all-if-ambiguous on")
    readline.parse_and_bind("tab: complete")
    if os.path.exists (historyPath) :
        readline.read_history_file(historyPath)
        readline.set_history_length(-1)


    ############################
    # Example code starts here #
    ############################

