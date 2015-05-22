from CMGTools.Production.ProductionTasks import Task
from CMGTools.Production.publish import publish
import os, sys

class PublishTask(Task):
    """Publish the dataset in DBS, Savannah"""
    def __init__(self, dataset, user, options):
        Task.__init__(self,'PublishTask', dataset, user, options)
        self.password = None
        self.development = options.development

    @staticmethod
    def addOptionStatic(parser):
        # This option will be used to find dataset on castor, and assign dataset
        parser.add_option("-F", "--fileown", 
                         dest="fileown",
                         help="User who is the files owner on EOS." ,
                         default=os.environ['USER'] )
        # If the purpose is to test the software use this parameter, it will not be recognised by the
        # non-testing algorithm
        parser.add_option("-T", "--test",
                         action = "store_true",
                         dest="test",
                         help="Flag task as a test",
                         default=False )
       # If user wants to add their own comments
        parser.add_option("-C", "--comment",
                         action = "store",
                         dest="commented",
                         help="Take comment as an argument",
                         default = None)

        # If user wants to add their own comments
        parser.add_option("-f", "--force",
                         action = "store_true",
                         dest="force",
                         help="force publish without logger",
                         default = False)
        # If user wants to add their own comments
        parser.add_option("-G", "--groups",
                         action = "store_true",
                         dest="checkGroups",
                         help="check the related group accounts on EOS",
                         default = False)
        # If user wants to publish primary dataset
        parser.add_option("-P", "--primary",
                         action = "store_true",
                         dest="primary",
                         help="publish a primary dataset",
                         default = False)

    def addOption(self,parser):
        self.addOptionStatic(parser)

    def run(self, input):
        username = os.getlogin()
        return publish(self.dataset,
                       self.options.fileown,
                       self.options.commented,
                       self.options.test,
                       username,
                       self.options.force,
                       self.options.primary,
                       (self.options.min_run, self.options.max_run), 
                       self.development )

