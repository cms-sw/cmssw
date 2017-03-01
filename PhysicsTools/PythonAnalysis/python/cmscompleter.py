"""Word completion for CMS Software.
Based on rlcompleter from Python 2.4.1. Included additional namespace for not loaded modules
Please read license in your lcg external python version

benedikt.hegner@cern.ch

"""
from __future__ import absolute_import
# TODO: sometimes results are doubled. clean global_matches list!

import readline
import rlcompleter
import __builtin__
import __main__

__all__ = ["CMSCompleter"]

class CMSCompleter(rlcompleter.Completer):
    def __init__(self, namespace = None):
	
        if namespace and not isinstance(namespace, dict):
            raise TypeError('namespace must be a dictionary')

        # Don't bind to namespace quite yet, but flag whether the user wants a
        # specific namespace or to use __main__.__dict__. This will allow us
        # to bind to __main__.__dict__ at completion time, not now.
        if namespace is None:
            self.use_main_ns = 1
        else:
            self.use_main_ns = 0
            self.namespace = namespace		
        try:
            # loading cms namespace
            from . import namespaceDict
            self.cmsnamespace = namespaceDict.getNamespaceDict()
        except:
            print 'Could not load CMS namespace'

 
    def global_matches(self, text):
        """Compute matches when text is a simple name.

        Return a list of all keywords, built-in functions and names currently
        defined in self.namespace and self.cmsnamespace that match.

        """
        import keyword
        matches = []
        n = len(text)
        for list in [keyword.kwlist,
                     __builtin__.__dict__,
					 self.cmsnamespace]:
            for word in list:
                if word[:n] == text and word != "__builtins__":
                    matches.append(word)
        return matches

# connect CMSCompleter with readline
readline.set_completer(CMSCompleter().complete)
readline.parse_and_bind('tab: complete')


