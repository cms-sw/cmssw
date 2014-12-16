import ROOT
def AddPhysObj(x):
	x.physObj = x
	return x
def AddPhysObjAndCallInit(x):
	x.physObj = x
	x._physObjInit()
	return x


def decorate(orig,deco):
   for b in deco.__bases__ :
        decorate(orig,b)
   for m in deco.__dict__ :
        if m[0] != "_" or m[1] != "_" :
                setattr(orig,m,deco.__dict__[m])
                #print m


#$ef decorate(oldstyle):
#   cmssw=eval("ROOT.pat.%s"%oldstyle.__name__)
#   for ty in tuple([oldstyle])+oldstyle.__bases__:
#  print ty
#      for m in ty.__dict__ :
#        if m[0] != "_" or m[1] != "_":
#                setattr(cmssw,m,ty.__dict__[m])
#               print m

#	decorate(o)
#	o=AddPhysObj
