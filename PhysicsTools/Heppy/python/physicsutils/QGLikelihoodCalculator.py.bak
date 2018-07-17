import ROOT
import math



def getBinNumber( bins, value ) :

  if value<bins[0] or value>bins[len(bins)-1]: return False
  for i in range(0,len(bins)-1):
    if value>bins[i] and value<bins[i+1]: 
      return i

  return -1




class QGLikelihoodCalculator:

  def __init__(self, filename) :
    self.pdfs = {}
    self.etaBins = []
    self.ptBins  = []
    self.rhoBins = []
    self.varNames = { 0:'mult', 1:'ptD', 2:'axis2' }
    self.qgNames = { 0:'quark', 1:'gluon' }
    self.init(filename)

  def init(self, filename) :

    print "[QGLikelihoodCalculator]: Initializing from file: " + filename

    f = ROOT.TFile.Open(filename)
    if f.IsZombie() : return False
    try :
      self.etaBins = f.Get("etaBins")
      self.ptBins  = f.Get("ptBins")
      self.rhoBins = f.Get("rhoBins")
    except :
      return False



    self.pdfs = {}
    
    for it in range(0,len(self.qgNames)):
      self.pdfs[self.qgNames[it]] = {}
      for iv in range(0,len(self.varNames)):
        self.pdfs[self.qgNames[it]][self.varNames[iv]] = {}
        for ie in range(0,len(self.etaBins)):
          self.pdfs[self.qgNames[it]][self.varNames[iv]][ie] = {}
          for ip in range(0,len(self.ptBins)):
            self.pdfs[self.qgNames[it]][self.varNames[iv]][ie][ip] = {}
            for ir in range(0,len(self.rhoBins)):
              self.pdfs[self.qgNames[it]][self.varNames[iv]][ie][ip][ir] = 0



    print "[QGLikelihoodCalculator]: Initialized binning of pdfs..."

    keys = f.GetListOfKeys()
    for key in keys :
      if key.IsFolder() == False: continue
      hists = key.ReadObj().GetListOfKeys()
      for hist in hists :
        pieces = hist.GetName().split("_")
        varName = pieces[0]
        qgType  = pieces[1]
        etaStr  = pieces[2]
        ptStr   = pieces[3]
        rhoStr  = pieces[4]
        etaBin  = int(etaStr.split("eta")[1])
        ptBin   = int( ptStr.split("pt") [1])
        rhoBin  = int(rhoStr.split("rho")[1])
        histogram = hist.ReadObj()
        histogram.SetDirectory(0)
        self.pdfs[qgType][varName][etaBin][ptBin][rhoBin] = histogram

    print "[QGLikelihoodCalculator]: pdfs initialized..."


    return True




  def isValidRange( self, pt, rho, eta ) :

    if pt < self.ptBins[0]: return False
    if pt > self.ptBins[len(self.ptBins)-1]: return False
    if rho < self.rhoBins[0]: return False
    if rho > self.rhoBins[len(self.rhoBins)-1]: return False
    if math.fabs(eta) < self.etaBins[0]: return False
    if math.fabs(eta) > self.etaBins[len(self.etaBins)-1]: return False
    return True




  def findEntry( self, eta, pt, rho, qgType, varName ) :

    etaBin = getBinNumber( self.etaBins, math.fabs(eta))
    if etaBin==-1: return None
    ptBin = getBinNumber( self.ptBins, pt )
    if ptBin==-1 : return None

    rhoBin = getBinNumber( self.rhoBins, rho )
    if rhoBin==-1 : return None

    return self.pdfs[qgType][varName][etaBin][ptBin][rhoBin]





  def computeQGLikelihood( self, jet, rho ):

    if self.isValidRange(jet.pt(), rho, jet.eta())==False:  return -1

    # careful!!! this needs to be in the same order of self.varNames
    vars = {0:jet.mult, 1:jet.ptd, 2:jet.axis2}

    Q=1.
    G=1.

    #print "----------------------"
    #if jet.partonFlavour()==21 :
    #  print "this jet is a GLUON"
    #elif math.fabs(jet.partonFlavour())<4 and jet.partonFlavour()!=0:
    #  print "this jet is a QUARK"
    #print "pt: " + str(jet.pt()) + " eta: " + str(jet.eta()) + " rho: " + str(rho)
    #print "multi: " + str(jet.mult) + " ptd: " + str(jet.ptd) + " axis2: " + str(jet.axis2)

    for i in vars :

      #print self.varNames[i] + ": " + str(vars[i])
      # quarks 
      qgEntry = self.findEntry(jet.eta(), jet.pt(), rho, "quark", self.varNames[i])

      if qgEntry == None:  return -1
      Qi = qgEntry.GetBinContent(qgEntry.FindBin(vars[i]))
      mQ = qgEntry.GetMean()
      #print "Qi: " + str(Qi)

      # gluons 
      qgEntry = self.findEntry(jet.eta(), jet.pt(), rho, "gluon", self.varNames[i])

      if qgEntry == None: return -1
      Gi = qgEntry.GetBinContent(qgEntry.FindBin(vars[i]))
      mG = qgEntry.GetMean()
      #print "Gi: " + str(Gi)

      epsilon=0.
      delta=0.000001
      if Qi <= epsilon and Gi <= epsilon :
        if mQ>mG :
          if vars[i] > mQ : 
             Qi = 1.-delta  
             Gi = delta
          elif vars[i] < mG : 
             Qi = delta
             Gi = 1.-delta
        else :
          if vars[i]<mQ :
             Qi = 1.-delta
             Gi = delta
          elif vars[i]>mG :
             Qi = delta
             Gi = 1.-delta

      Q*=Qi
      G*=Gi
     

    #print "Q: " + str(Q)
    #print "G: " + str(G)
    if Q==0. : return 0.
    else : return Q/(Q+G)

