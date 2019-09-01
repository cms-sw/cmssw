## @package CrabConfigParser
# This module extends the python configparser to create crab3 config files
#
# This module extends the python configparser to create crab3 config files.


from ConfigParser import *

## The CrabConfigParser class
#
# This class extends the python ConfigParser class and adds functions to
# output crab3 config files
class CrabConfigParser(ConfigParser):

    ## The constructor.
    def __init__(self):
        ConfigParser.__init__(self)
        self.optionxform = str
    ## Write CrabConfigParser object to file
    # @type self: CrabConfigParser
    # @param self: The object pointer.
    # @type filename: string
    # @param filename The name of the output crab3 config file.
    def writeCrabConfig(self,filename):
        sections = self.sections()
        fixedsections = ['General','JobType','Data','Site','User','Debug']
        outlines = []
        # Add inital header for crab config file
        outlines.append('from WMCore.Configuration import Configuration \n')
        outlines.append('config = Configuration()')
        # we will first add the main crab3 config sections in the given order
        for fixedsection in fixedsections:
            if fixedsection in sections:
                outlines.extend(self.getSectionLines(fixedsection))
                sections.remove(fixedsection)
        # add additional sections (may be added in future crab3 versions ?)
        for section in sections:
            outlines.extend(self.getSectionLines(section))
        #print filename
        with open(filename,'wb') as outfile:
            outfile.write('\n'.join(outlines) + '\n')
    ## Helper function to retrieve crab config output lines for one section
    # @type self: CrabConfigParser
    # @param self:The object pointer.
    # @type section: string
    # @param section:The section name.
    # @rtype: list of strings
    # @return: Lines for one section in crab3 config file
    def getSectionLines(self,section):
        sectionLines = []
        sectionLines.append('\nconfig.section_("%s")'%section)
        configItems =  self.items(section)
        for configItem in configItems:
            if not isinstance(configItem[1], str):
                sectionLines.append('config.%s.%s = %s'%(section,configItem[0],configItem[1]))
            elif "True" in configItem[1] or "False" in configItem[1]:
                sectionLines.append('config.%s.%s = %s'%(section,configItem[0],configItem[1]))
            else:
                parsed = False
                if configItem[0]=="runRange" :
                    sectionLines.append('config.%s.%s = \'%s\''%(section,configItem[0],configItem[1]))
                    parsed = True
                if not parsed:
                    try:
                        sectionLines.append('config.%s.%s = %d'%(section,configItem[0],int(configItem[1])))
                        parsed = True
                    except:
                        pass
                if not parsed:
                    try:
                        sectionLines.append('config.%s.%s = %.2f'%(section,configItem[0],float(configItem[1])))
                        parsed = True
                    except:
                        pass
                if not parsed:
                    if isinstance(configItem[1], list):
                        sectionLines.append('config.%s.%s = %s'%(section,configItem[0],str(configItem[1])))
                        parsed = True
                if not parsed:
                    sectionLines.append('config.%s.%s = \'%s\''%(section,configItem[0],configItem[1]))

        return sectionLines
