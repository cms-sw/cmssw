"""

This file is for functions that can be used to build propositional logic trees for query construction

"""

def eq(a, b):
	return str(a) + "=" + ("'%s'" % b if type(b) == str else str(b))