import logging

from PyQt4.QtCore import SIGNAL,Qt,QCoreApplication
from PyQt4.QtGui import QWidget

from Vispa.Views.AbstractView import AbstractView
from Vispa.Main.Exceptions import exception_traceback
from Vispa.Share.BasicDataAccessor import BasicDataAccessor

try:
    import ROOT
    import pxl.core,pxl.astro,pxl.hep
    import_root_error=None
except Exception,e:
    import_root_error=(str(e),exception_traceback())

from array import array

class RootCanvasView(AbstractView, QWidget):
    
    LABEL = "&ROOT Canvas View"
    
    def __init__(self, parent=None):
        AbstractView.__init__(self)
        QWidget.__init__(self, parent)
        ROOT.gROOT.SetStyle('Plain')
        #ROOT.gStyle.SetPalette(1)
        self.canvas = ROOT.TCanvas()
        #self.canvas.SetEditable(False)
        #self.canvas = None 
        self._operationId = 0
        self._updatingFlag = 0

    def setDataAccessor(self, accessor):
        """ Sets the DataAccessor from which the data is read 
        You need to call updateContent() in order to make the changes visible.   
        """
        if not isinstance(accessor, BasicDataAccessor):
            raise TypeError(__name__ + " requires data accessor of type BasicDataAccessor.")
        #if not isinstance(accessor, RelativeDataAccessor):
        #    raise TypeError(__name__ + " requires data accessor of type RelativeDataAccessor.")
        AbstractView.setDataAccessor(self, accessor)
        
    def updateContent(self):
        """ Clear the view and refill it.
        """
        logging.debug(__name__ + ": updateContent")
        if import_root_error!=None:
            logging.error(__name__ + ": Could not import pxl and ROOT: "+import_root_error[1])
            QCoreApplication.instance().errorMessage("Could not import pxl and ROOT (see logfile for details):\n"+import_root_error[0])
            return
        self._updatingFlag +=1
        operationId = self._operationId
        #if self._dataAccessor:
        #objects = self._filter(self._dataObjects)
        objects = self.applyFilter(self.dataObjects())
        i = 0
        for object in objects:
            if operationId != self._operationId:
                break
            self._plotscript(self.canvas, object) 

        self._updatingFlag -=1
        return operationId == self._operationId
  
    def _plotscript(self, canvas, object):
      ''' The actual plotting script - has to be replaced later with
      script from external file by user'''
      
      
      logging.debug(__name__ + ": _plotscript")
      canvas.cd()
      if isinstance(object, pxl.core.BasicContainer):
        self.basiccontainer = object
        abos = object.getObjectsOfType(pxl.astro.AstroBasicObject)
        logging.debug(__name__ + ": _plotscript: Plotting " + str(len(abos)) + " AstroObjects")
        lat = []
        lon = []
        for ao in abos:
          lat.append(ao.getLatitude())
          lon.append(ao.getLongitude())
        
        self.f = ROOT.TGraph(len(lat), array('f', lon), array('f', lat))
        #self.f.SetEditable(False) 
        self.f.SetTitle('')
        self.f.GetXaxis().SetTitle('Longitude')
        self.f.GetYaxis().SetTitle('Latitude')
        self.f.SetName('')
        self.f.Draw('ap')
        self.f.SetEditable(False)
        self.p = None
      elif isinstance(object, pxl.astro.RegionOfInterest) or isinstance(object, pxl.astro.UHECRSource):
        self.p = []
        self.p.append(ROOT.TMarker(object.getLongitude(), object.getLatitude(), 3))
        logging.debug(__name__ + 'DrawROI with ' + str(len(object.getSoftRelations().getContainer().items())) + ' UHECR')
        for key, item in object.getSoftRelations().getContainer().items():
          
          uhecr = pxl.astro.toUHECR(self.basiccontainer.getById(item))
          if isinstance(uhecr,pxl.astro.UHECR):
            self.p.append(ROOT.TMarker(uhecr.getLongitude(), uhecr.getLatitude(), 7))
            self.p[ - 1].SetMarkerSize(10)
            self.p[ - 1].SetMarkerColor(ROOT.kGreen + 1)
          else:
            uhecr = pxl.astro.toAstroObject(self.basiccontainer.getById(item))
            if isinstance(uhecr,pxl.astro.AstroObject):
              self.p.append(ROOT.TMarker(uhecr.getLongitude(), uhecr.getLatitude(), 7))
              self.p[ - 1].SetMarkerSize(8)
              self.p[ - 1].SetMarkerColor(ROOT.kOrange + 1)

        for x in self.p:
          x.Draw()
      
      elif isinstance(object, pxl.astro.AstroBasicObject):
        self.p = ROOT.TMarker(object.getLongitude(), object.getLatitude(), 7)
        self.p.SetMarkerSize(10)
        self.p.SetMarkerColor(ROOT.kRed)
        self.p.Draw()
      elif isinstance(object, pxl.hep.EventView):
        self.createLegoPlot(object)
      elif isinstance(object, pxl.core.Event):
        for eventview in object.getObjectsOfType(pxl.hep.EventView):
          self.createLegoPlot(object)
      elif isinstance(object, pxl.hep.Particle):
        self.p = ROOT.TMarker(object.getPhi(), object.getEta(), 20)
        if (object.getName() == "Muon"):
          h2 = ROOT.TH2F('muon-plot', '', 50, - 4, 4, 50, - 3.141593, 3.141593)
          h2.SetFillColor(2)
          h2.Fill(object.getEta(), object.getPhi(), object.getPt())
          h2.DrawCopy('SAME LEGO A')
        elif (object.getName() == "Jet"):
          h2 = ROOT.TH2F('jet-plot', '', 50, - 4, 4, 50, - 3.141593, 3.141593)
          h2.SetFillColor(4)
          h2.Fill(object.getEta(), object.getPhi(), object.getPt())
          h2.DrawCopy('SAME LEGO A')
        elif (object.getName() == "Electron"):
          h2 = ROOT.TH2F('electron-plot', '', 50, - 4, 4, 50, - 3.141593, 3.141593)
          h2.SetFillColor(3)
          h2.Fill(object.getEta(), object.getPhi(), object.getPt())
          h2.DrawCopy('SAME LEGO A')
        elif (object.getName() == "MET"):
          h2 = ROOT.TH2F('met-plot', '', 50, - 4, 4, 50, - 3.141593, 3.141593)
          h2.SetFillColor(6)
          h2.Fill(object.getEta(), object.getPhi(), object.getPt())
          h2.DrawCopy('SAME LEGO A')
        else:
          h2 = ROOT.TH2F('all-plot', '', 50, - 4, 4, 50, - 3.141593, 3.141593)
          h2.SetFillColor(1)
          h2.Fill(object.getEta(), object.getPhi(), object.getPt())
          h2.DrawCopy('SAME LEGO A')
        self.p.Draw()
        
      canvas.Modified()
      canvas.Update()

    def createGraph(self, eventview):
      particles = eventview.getObjectsOfType(pxl.hep.Particle)
      logging.info("Display " + str(len(particles)) + " Particles.")
      if (len(particles) > 0):
        etas = []
        phis = []
        for particle in particles:
          if (particle.getEta() < 1000.):
            etas.append(particle.getEta())
            phis.append(particle.getPhi())
        self.f = ROOT.TGraph(len(etas), array('f', phis), array('f', etas))
        self.f.SetTitle('')
        self.f.GetXaxis().SetTitle('#Phi')
        self.f.GetXaxis().SetRangeUser(- 3.14156, 3.14156)
        self.f.GetYaxis().SetTitle('#eta')
        self.f.GetXaxis().SetRangeUser(- 5., 5.)
        self.f.SetMarkerSize(1)
        self.f.SetMarkerStyle(20)
        self.f.SetName('')
        self.f.Draw('ap')
        self.f.SetEditable(False)
        self.p = None

    def createLegoPlot(self, eventview):
      particles = eventview.getObjectsOfType(pxl.hep.Particle)
      logging.info("Display " + str(len(particles)) + " Particles")
      if (len(particles) > 0):
        self.f = ROOT.TH2F('eta-phi-plot', '', 50, - 4, 4, 50, - 3.141593, 3.141593)
        for particle in particles:
          self.f.Fill(particle.getEta(), particle.getPhi(), particle.getPt())

        self.f.SetTitle('')
        self.f.GetXaxis().SetTitle('#eta')
        self.f.GetYaxis().SetTitle('#Phi')
        self.f.GetZaxis().SetTitle('p_{T} (GeV)')
        self.f.SetName('')
        self.f.DrawCopy('LEGO2')
        self.p = None
    
    def cancel(self):
        """ Stop all running operations.
        """
        self._operationId += 1

    def mousePressEvent(self,event):
        QWidget.mousePressEvent(self,event)
        if event.button()==Qt.RightButton:
            self.emit(SIGNAL("mouseRightPressed"), event.globalPos())

    def isBusy(self):
        return self._updatingFlag>0
