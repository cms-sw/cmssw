# example PatTool
#
# http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/PhysicsTools/PatAlgos/python/tools/coreTools.py?view=markup
#
# def removeMCMatching(process,
#                      name
#                      ):
#     """
#     ------------------------------------------------------------------
#     remove monte carlo matching from a given collection or all PAT
#     candidate collections:
#
#     process : process
#     name    : collection name; supported are 'Photons', 'Electrons',
#               'Muons', 'Taus', 'Jets', 'METs', 'All'
#     ------------------------------------------------------------------    
#     """

class RemoveMCMatching(ActionInterface):
    def __init__(self):
        self._parameters=[\
            ("String","name","All","collection name; supported are 'Photons', 'Electrons','Muons', 'Taus', 'Jets', 'METs', 'All'")\
                          ]
        
    def dumpPython(self):
        """ Return the python code to perform the action.
        """
        return "removeMCMatching(process, "+self._parameters[0]+")"

    def label(self):
        """ Return the label of the action.
        """
        return "removeMCMatching"
    def description(self):
        """ Return a string with a detailed description of the action.
        """
        return "remove monte carlo matching from a given collection or all PAT candidate collections"
    def parameters(self):
        """ Return the list of the parameters of an action.
        
        Each parameters is represented by a tuple containing its
        type, name, value and description.
        The type determines how the parameter is represented in the GUI.
        Possible types are: 'Category','String','Text','File','FileVector','Boolean','Integer','Float'.
        """
        return self._parameters
    def setParameter(self, name, value):
        """ Change the parameter 'name' to a new value.
        """
        for p in self.parameters():
            if name==p[1]:
                p[2]=value
        else:
            raise NameError("parameter "+name+" unkown.")
