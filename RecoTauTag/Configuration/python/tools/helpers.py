import FWCore.ParameterSet.Config as cms

## Helpers to perform some technically boring tasks like looking for all modules with a given parameter
## and replacing that to a given value

# Next two lines are for backward compatibility, the imported functions and
# classes used to be defined in this file.
from FWCore.ParameterSet.MassReplace import massSearchReplaceAnyInputTag, MassSearchReplaceAnyInputTagVisitor
from FWCore.ParameterSet.MassReplace import massSearchReplaceParam, MassSearchParamVisitor, MassSearchReplaceParamVisitor

class CloneTaskVisitor(object):
    """Visitor that travels within a cms.Task, and returns a cloned version of the Task.
    All modules are cloned and a postfix is added"""
    def __init__(self, process, label, postfix, removePostfix="", noClones = [], verbose = False):
        self._process = process
        self._postfix = postfix
        self._removePostfix = removePostfix
        self._noClones = noClones
        self._verbose = verbose
        self._moduleLabels = []
        self._clonedTask = cms.Task()
        setattr(process, self._newLabel(label), self._clonedTask)

    def enter(self, visitee):
        if isinstance(visitee, cms._Module):
            label = visitee.label()
            newModule = None
            if label in self._noClones: #keep unchanged
                newModule = getattr(self._process, label)
            elif label in self._moduleLabels: # has the module already been cloned ?
                newModule = getattr(self._process, self._newLabel(label))
            else:
                self._moduleLabels.append(label)
                newModule = visitee.clone()
                setattr(self._process, self._newLabel(label), newModule)
            self.__appendToTopTask(newModule)

    def leave(self, visitee):
        pass

    def clonedTask(self):#FIXME: can the following be used for Task?
        for label in self._moduleLabels:
            massSearchReplaceAnyInputTag(self._clonedTask, label, self._newLabel(label), moduleLabelOnly=True, verbose=self._verbose)
        self._moduleLabels = [] # prevent the InputTag replacement next time the 'clonedTask' function is called.
        return self._clonedTask

    def _newLabel(self, label):
        if self._removePostfix != "":
            if label[-len(self._removePostfix):] == self._removePostfix:
                label = label[0:-len(self._removePostfix)]
            else:
                raise Exception("Tried to remove postfix %s from label %s, but it wasn't there" % (self._removePostfix, label))
        return label + self._postfix

    def __appendToTopTask(self, visitee):
        self._clonedTask.add(visitee)

def cloneProcessingSnippetTask(process, task, postfix, removePostfix="", noClones = [], verbose = False):
    """
    ------------------------------------------------------------------
    copy a task plus the modules and tasks therein
    both are renamed by getting a postfix
    input tags are automatically adjusted
    ------------------------------------------------------------------
    """
    result = task
    if not postfix == "":
        visitor = CloneTaskVisitor(process, task.label(), postfix, removePostfix, noClones, verbose)
        task.visit(visitor)
        result = visitor.clonedTask()
    return result

