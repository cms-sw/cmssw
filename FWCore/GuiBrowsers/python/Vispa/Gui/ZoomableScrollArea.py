from PyQt4.QtCore import *
from PyQt4.QtGui import *

from Vispa.Gui.Zoomable import Zoomable
from Vispa.Gui.ZoomableWidget import ZoomableWidget

class ZoomableScrollArea(Zoomable, QScrollArea):
	""" Standard QScrollArea extended by zooming capabilties.
	"""
	
	def __init__(self, parent=None):
		""" Constructor.
		"""
		QScrollArea.__init__(self, parent)
		Zoomable.__init__(self)               # Call after QScrollArea constructor, required by setZoom()
		
	def wheelEvent(self, event):
		""" If wheelEvent occurs either zoom window (ctrl-key pressed) or scroll the area.
		"""
		if event.modifiers() == Qt.ControlModifier:
			oldZoom = self.zoom()
			oldLeft = self.widget().mapFrom(self,QPoint(event.x(),0)).x()
			oldTop = self.widget().mapFrom(self,QPoint(0,event.y())).y()
			
			if event.delta() > 0:
				self.incrementZoom()
			else:
				self.decrementZoom()
				
			newZoom = self.zoom()
			zoomFactor = newZoom / oldZoom
			newLeft = oldLeft * zoomFactor 
			newTop = oldTop * zoomFactor

			self.autosizeScrollWidget()
			self.ensureVisible(newLeft-event.x()+self.viewport().width()/2.0,newTop-event.y()+self.viewport().height()/2.0,self.viewport().width()/2.0,self.viewport().height()/2.0)
			self.emit(SIGNAL('wheelZoom()'))
		else:
			QScrollArea.wheelEvent(self, event)

	def setZoom(self, zoom):
		""" Sets its own zoom factor and passes it to it's child widget if child is Zoomable.
		"""
		Zoomable.setZoom(self, zoom)
		if isinstance(self.widget(), ZoomableWidget):
			self.widget().setZoom(zoom)
		self.autosizeScrollWidget()
		self.emit(SIGNAL("zoomChanged(float)"), zoom)
	
	def resizeEvent(self, event):
		"""Calls autosizeScrollWidget().
		"""
		self.autosizeScrollWidget()
		QScrollArea.resizeEvent(self, event)
			
	def autosizeScrollWidget(self):
		"""Sets size of child widget to the size needed to fit whole content.
		"""
		if not self.widget():
			return
		childrenRect = self.widget().childrenRect()
		width = max(self.viewport().width(), childrenRect.bottomRight().x()) - min(0, childrenRect.topLeft().x())
		height = max(self.viewport().height(), childrenRect.bottomRight().y()) - min(0, childrenRect.topLeft().y())
 		self.widget().resize(width, height)
 		
 	def mousePressEvent(self, event):
 		""" Forward mousePressEvent.
 		"""
	 	self.widget().mousePressEvent(event)
