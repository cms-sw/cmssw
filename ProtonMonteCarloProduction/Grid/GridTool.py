import os 
os.system("source /cvmfs/cms.cern.ch/crab3/crab.sh")

from cmd import Cmd
from crabutil import colors
from crabutil import dataset_parser
from crabutil import crab_library
from optparse import OptionParser
from CRABClient.UserUtilities import config, getUsernameFromSiteDB

def getOptions():

    """
    Parse and return the arguments provided by the user.
    """

    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--filename",
                  metavar="FILE", help="XML mapping file", default='grid_samples_2017_toy_muon_step0.xml')
    parser.add_option("-p", "--parsing",
                  help="parsing: commands which can be passed from SHELL directly. [parsing: --p \"submit --file filename.xml\"]")

    parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="make lots of noise [default]")

    (options, arguments) = parser.parse_args()
    return options

def execute(fromXML, crabsub, config):

     #try:
                for i in fromXML:
                                #(dataset, mode, era, year, xangle, mass, configfile, nevents, with_dataset, tagname)
                                crabsample = crabsub.doSubmit(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11], config)

     #except:
     #           print color.FAIL+color.BOLD+'\tFailed! Please try again!'+color.ENDC+color.HEADER+color.ENDC
     #           exit(0)

class MyPrompt(Cmd):
    prompt = 'crab_submission> '
    intro = "Type ? to list commands"
 
    def do_exit(self, inp):
        print("\n Exiting...\n")
        return True
    
    def help_exit(self):
        print('exit the application. Shorthand: x q Ctrl-D.')
 
    def do_submit(self, arg):

	arcmd = arg.split()
	filename = options.filename

	if ("--file" or "--f") in arg: 
                if len(arcmd)>1:
                        for i in arcmd[1:]:
                                filename=str(i)

		if os.path.exists(filename):
	        	xml = dataset_parser.Parser(filename, options.verbose)
	        	fromXML = xml.GetFromXML(options.verbose)
		        crabsub = crab_library.CrabLibrary()
			execute(fromXML, crabsub, config)
		
		else:
        	        print color.FAIL+color.BOLD+'\tXML file does not exist or wrong path! Please use the option --file filename.xml or run the application again with the option --f filename.xml'+color.ENDC+color.HEADER+color.ENDC

	else:
                xml = dataset_parser.Parser(options.filename, options.verbose)
                fromXML = xml.GetFromXML(options.verbose)
                crabsub = crab_library.CrabLibrary()
                execute(fromXML, crabsub, config)			
	
    def help_submit(self):
        print("\n\nUse the options --file filename.xml\n\tEx: submit --file filename.xml <press enter>\n\n")

    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)
 
        print("Default: {}".format(inp))

    def emptyline(self):
        pass

    do_EOF = do_exit
    help_EOF = help_exit
 
if __name__ == '__main__':

    color = colors.Paint()
    options = getOptions()

    config = config()
    if options.parsing:
	MyPrompt().onecmd(''.join(options.parsing))
    else:
        MyPrompt().cmdloop()

