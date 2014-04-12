import logging

from PyQt4.QtCore import Qt, QPoint, QPointF, QRectF, QSizeF, SIGNAL
from PyQt4.QtGui import QColor, QPainter, QPen, QLinearGradient, QRadialGradient, QPainterPath

from Vispa.Gui.ZoomableWidget import ZoomableWidget
from Vispa.Gui.VispaWidgetOwner import VispaWidgetOwner

class PointToPointConnection(ZoomableWidget):
    """ Visualizes a connection between two points.
    
    There are several types of connections available (see setType()):
       ORTHOGONAL    - Start and end points are connected by a set of horizontal and vertical lines (default).
       STRAIGHT      - Start and end points are connected by a straight line. 
       DIAGONAL      - Start and end points are connected by a set of three diagonal lines with different angles.
    """    
    CONNECTOR_LENGTH = 15   # Length in pixels of the start/end part of the connection in direction of the startOrientation.
    CONNECTION_THICKNESS = 5
    FILL_COLOR1 = QColor(155, 0, 0)   # border
    FILL_COLOR2 = QColor(255, 0, 0)   # center
    
    SELECT_COLOR = QColor('darkblue')
    SELECTED_FRAME_WIDTH = 4            # Width in pixels of colored (SELECT_CORLOR) frame, when selected
    
    FOCUSPOLICY = Qt.ClickFocus
    
    CONNECTION_TYPE = "ORTHOGONAL"
    
    class ConnectionDirection:
        UNDEFINED = 0
        UP = 1
        LEFT = 2
        DOWN = 3
        RIGHT = 4
    
    class DrawOrientation:
        HORIZONTAL = 0
        VERTICAL = 1
        
    class CornerType:
        UNDEFINED = 0
        TOP_RIGHT = 1
        TOP_LEFT = 2
        BOTTOM_LEFT = 3
        BOTTOM_RIGHT = 4
        STRAIGHT = 5
        
    def __init__(self, workspace, sourcePoint, targetPoint):
        #logging.debug(__name__ +": __init__()")
        self._recalculateRouteFlag = True
        self._sourceDirection = PointToPointConnection.ConnectionDirection.RIGHT
        self._targetDirection = PointToPointConnection.ConnectionDirection.LEFT
        self._route = None
        self._selectableFlag = True
        self._selectedFlag = False
        self._deletableFlag = True
        self._deletedFlag = False
        self._sourcePoint = sourcePoint
        self._targetPoint = targetPoint
        self._dragReferencePoint = None
        
        ZoomableWidget.__init__(self, workspace)
        self.setFocusPolicy(self.FOCUSPOLICY)
        self.setType(self.CONNECTION_TYPE)
        
        self.updateConnection()
        
        # debug
        #self.setAutoFillBackground(True)
        #self.setPalette(QPalette(Qt.green))
        
    def setType(self, type):
        """ Sets type of connection.
        
        The type argument is a string of capitalized letters and should be one of the available types described in the class documentation.
        """
        self._type = type
    
    def routeChanged(self):
        self._recalculateRouteFlag = True

    def setZoom(self, zoom):
        ZoomableWidget.setZoom(self, zoom)
        self.forceRouteRecalculation()
        
    def sourcePoint(self):
        return self._sourcePoint
    
    def targetPoint(self):
        return self._targetPoint
    
    def updateTargetPoint(self, point):
        self._targetPoint = point
        self.updateConnection()
    
    def setSourceDirection(self, dir):
        self._sourceDirection = dir
    
    def sourceDirection(self):
        return self._sourceDirection
    
    def setTargetDirection(self, dir):
        self._targetDirection = dir
        
    def targetDirection(self):
        return self._targetDirection

    def setSelectable(self, sel):
        self._selectableFlag = sel
    
    def isSelectable(self):
        return bool(self._selectableFlag)
    
    def setDeletable(self, sel):
        self._deletableFlag = sel
    
    def isDeletable(self):
        return bool(self._deletableFlag)
    
    def isSelected(self):
        return self._selectedFlag
    
    def select(self, sel=True, multiSelect=False):
        if not self.isSelectable():
            return
        changed = False
        if self._selectedFlag != sel:
            changed = True
            
        self._selectedFlag = sel

        if self._selectedFlag:
            self.raise_()
            
        if changed:
            self.update()
        
        if not multiSelect and self.isSelected() and isinstance(self.parent(), VispaWidgetOwner):
            self.parent().deselectAllWidgets(self)
    
    def setDragReferencePoint(self, pos):
        self._dragReferencePoint = pos
        
    def dragReferencePoint(self):
        return self._dragReferencePoint
        
    def cornerTypeString(self, type):
        if type == self.CornerType.TOP_RIGHT:
            return "TOP_RIGHT"
        elif type == self.CornerType.TOP_LEFT:
            return "TOP_LEFT"
        elif type == self.CornerType.BOTTOM_LEFT:
            return "BOTTOM_LEFT"
        elif type == self.CornerType.BOTTOM_RIGHT:
            return "BOTTOM_RIGHT"
        elif type == self.CornerType.STRAIGHT:
            return "STRAIGHT"
        return "UDEFINED_CORNER"
            
    def connectionDirectionString(self, dir):
        if dir == self.ConnectionDirection.DOWN:
            return "DOWN"
        elif dir == self.ConnectionDirection.LEFT:
            return "LEFT"
        elif dir == self.ConnectionDirection.RIGHT:
            return "RIGHT"
        elif dir == self.ConnectionDirection.UP:
            return "UP"
        return "UNDEFINED_DIRECTION"

    def forceRouteRecalculation(self):
        self._recalculateRouteFlag = True

    def routeIsValid(self):
        try:
            if type(self._route).__name__ != 'list':
                logging.error("PointToPointConnection.routeIsValid() - 'route' does not contain a list an so is not a valid route.")
                return False
        except:
            logging.error("PointToPointConnection.routeIsValid() - 'route' is not valid.")
            return False
        return True
    
    def betweenTwoPoints(self, point, first, second):
        """ Checks whether 'point' lies between 'first' and 'second'.
        
        This function can currently (08-11-15) only deal with horizontal and vertical distances.
        """
        
        halfthick = 0.5 * self.CONNECTION_THICKNESS * self.zoomFactor() 
        direction = self.getPointToPointDirection(first, second)
        
        topLeft = QPointF()
        bottomRight = QPointF()
        if direction == self.ConnectionDirection.LEFT:
            # horizontal, negative
            topLeft.setX(second.x())
            topLeft.setY(second.y() - halfthick)
            bottomRight.setX(first.x())
            bottomRight.setY(first.y() + halfthick)
        elif direction == self.ConnectionDirection.RIGHT:
            # horizontal, positive
            topLeft.setX(first.x())
            topLeft.setY(first.y() - halfthick)
            bottomRight.setX(second.x())
            bottomRight.setY(second.y() + halfthick)
        elif direction == self.ConnectionDirection.UP:
            # vertical, negative
            topLeft.setX(second.x() - halfthick)
            topLeft.setY(second.y())
            bottomRight.setX(first.x() + halfthick)
            bottomRight.setY(first.y())
        elif direction == self.ConnectionDirection.UP:
            # vertical, positive
            topLeft.setX(first.x() - halfthick)
            topLeft.setY(first.y())
            bottomRight.setX(second.x() + halfthick)
            bottomRight.setY(second.y())
        else:
            return False
        
        rect = QRectF(topLeft, bottomRight)
        return rect.contains(QPointF(point))

    def belongsToRoute(self, point):
        """ Checks whether 'point' is part of the connection.
        
        'point' has to be in coordinates of self.drawPanel.
        """  
        
        if not self.routeIsValid():
            return False
        
        lastP = None
        for thisP in self._route:
            if lastP != None:
                if self.getRectBetweenTwoPoints(lastP, thisP).contains(QPointF(point)):
                    return True
            lastP = thisP

        return False

    def getPointByDistance(self, start, distance, direction):
        """ Returns a point which is about 'distance' pixels remotely from 'start' in direction 'direction'.
        """
        
        if direction == self.ConnectionDirection.DOWN:
            return QPoint(start.x(), start.y() + distance)
        elif direction == self.ConnectionDirection.LEFT:
            return QPoint(start.x() - distance, start.y())
        elif direction == self.ConnectionDirection.RIGHT:
            return QPoint(start.x() + distance, start.y())
        elif direction == self.ConnectionDirection.UP:
            return QPoint(start.x(), start.y() - distance)
        else:
            logging.error("PointToPointConnection.getPointByDistance() - Unknown ConnectionDirection.")
            
    def nextPointByDistance(self, route, distance, direction):
        """ Directly adds getPointByDistance() to 'route'.
        """
        
        if len(route) < 1:
            logging.error("PointToPointConnection.nextPointByDistance() - Can not calculate next point for empty route.")
            return
        
        start = route[len(route) - 1]
        return self.getPointByDistance(start, distance, direction)
    
    def nextPointByTarget(self, route, target, orientation):
        """ Adds a point to 'route', so the route approaches to the 'target' point in a specific 'orientation'.
        
        This means that the added point and 'target' have one coordinate in common.
        If 'orientation' points in horizontal direction (left or right) the x components will be equal.
        If 'orientation' points in vertical direction (up or down) the y components will be equal.
        """
        if len(route) < 1:
            logging.error("PointToPointConnection.nextPointByTarget() - Can not calculate next point for empty route.")
            return
        
        start = route[len(route) - 1]
        
        if orientation == self.DrawOrientation.HORIZONTAL:
            return QPoint(target.x(), start.y())
        elif orientation == self.DrawOrientation.VERTICAL:
            return QPoint(start.x(), target.y())
        else:
            logging.error("PointToPointConnection.nextPointByTarget() - Unknown DrwaOrientation.")
    
    def calculateRoute(self):
        """ Calculates the route and stores all route points in internal list.
        
        If start and end points have not changed since last last calculation, the route wont be recalculated unless forceRouteRecalculation() was called before.
        If route was recalculated function returns True otherwise False.
        """
        if not self._recalculateRouteFlag and self._route != None and len(self._route) > 1:
            if self._route[0] == self.sourcePoint() and self._route[len(self._route) - 1] == self.targetPoint():
                # Nothing has changed, so route not recalculated
                return False
        
        # Recaclucating route
        
        # Whenever the start directions point at each other, the connection has to go through the center of the points
        throughCenter = False
        rowKind = False
        columnKind = False
        
        sourceDirection = self.sourceDirection()
        targetDirection = self.targetDirection()
        
        if self._type == "ORTHOGONAL":
            if sourceDirection == self.ConnectionDirection.RIGHT and targetDirection == self.ConnectionDirection.LEFT:
                throughCenter = True
                rowKind = True
            elif sourceDirection == self.ConnectionDirection.LEFT and targetDirection == self.ConnectionDirection.RIGHT:
                throughCenter = True
                rowKind = True
            elif sourceDirection == self.ConnectionDirection.DOWN and targetDirection == self.ConnectionDirection.UP:
                throughCenter = True
                columnKind = True
            elif sourceDirection == self.ConnectionDirection.UP and targetDirection == self.ConnectionDirection.DOWN:
                throughCenter = True
                columnKind = True
        
        self._route = []
        
        sP = QPoint(self.sourcePoint())        # start
        eP = QPoint(self.targetPoint())          # end
        self._route.append(sP)
        
        
        if throughCenter:
            # ORTHOGONAL
            centerP = (sP + eP) * 0.5
            firstP = self.nextPointByDistance(self._route, self.CONNECTOR_LENGTH * self.zoomFactor(), sourceDirection)
            lastP = self.getPointByDistance(eP, self.CONNECTOR_LENGTH * self.zoomFactor() , targetDirection)
            self._route.append(firstP)
            if rowKind:
                #if lastP.x() - firstP.x() > self.CONNECTOR_LENGTH * self.zoomFactor() * 0.5:
                if eP.x() - sP.x() > (self.CONNECTOR_LENGTH +1) * self.zoomFactor() * 2:
                    self._route.append(self.nextPointByTarget(self._route, centerP, self.DrawOrientation.HORIZONTAL))
                    #self._route.append(centerP)
                    self._route.append(self.nextPointByTarget(self._route, lastP, self.DrawOrientation.VERTICAL))
                else:
                    self._route.append(self.nextPointByTarget(self._route, centerP, self.DrawOrientation.VERTICAL))
                    #self._route.append(centerP)
                    self._route.append(self.nextPointByTarget(self._route, lastP +QPoint(-self.CONNECTOR_LENGTH * self.zoomFactor(), 0), self.DrawOrientation.HORIZONTAL))
                    #self._route.append(self.nextPointByDistance(self._route, self.CONNECTOR_LENGTH * self.zoomFactor() , self.targetDirection()))
                    self._route.append(self.nextPointByTarget(self._route, lastP, self.DrawOrientation.VERTICAL))
                    
            elif columnKind:
                #print "    columnKind"
                route.append(self.nextPointByTarget(self._route, centerP, self.DrawOrientation.VERTICAL))
                route.append(centerP)
                route.append(self.nextPointByTarget(self._route, lastP, self.DrawOrientation.HORIZONTAL))
            else:
                logging.error("PointToPointConnection.calculateRoute() - Sorry connections going through the center have to be either rowKind or columKind.")
                
            self._route.append(lastP)
        else:
            # STRAIGHT or DIAGONAL
            if self._type == "DIAGONAL":
                width = abs(sP.x() - eP.x())
                height = abs(sP.y() - eP.y())
                if width > 0:
                    directionX = (sP.x() - eP.x()) / width
                else:
                    directionX = 0
                if height > 0:
                    directionY = (sP.y() - eP.y()) / height
                else:
                    directionY = 0
                if width > height:
                    diagonal = height / 2
                else:
                    diagonal = width / 2
                self._route.append(QPoint(sP.x() - directionX * diagonal, sP.y() - directionY * diagonal))
                self._route.append(QPoint(sP.x() - directionX * (width - diagonal), sP.y() - directionY * (height - diagonal)))
        
        self._route.append(eP) 
        self._recalculateRouteFlag = False
        return True
    
    def getPointToPointDirection(self, lastP, thisP):
        
        if not lastP or not thisP:
            return self.ConnectionDirection.UNDEFINED
        
        if lastP.y() == thisP.y():    # horizontal
            if lastP.x() < thisP.x():
                return self.ConnectionDirection.RIGHT
            else:
                return self.ConnectionDirection.LEFT
        elif lastP.x() == thisP.x():    # vertical
            if lastP.y() < thisP.y():
                return self.ConnectionDirection.DOWN
            else:
                return self.ConnectionDirection.UP
                
        return self.ConnectionDirection.UNDEFINED

    def cornerIsDefined(self, type):
        if type == self.CornerType.TOP_RIGHT or type == self.CornerType.TOP_LEFT or type == self.CornerType.BOTTOM_LEFT or type == self.CornerType.BOTTOM_RIGHT:
            return True
        return False

    def getCornerType(self, lastDirection, thisDirection):
        if (lastDirection == self.ConnectionDirection.UP and thisDirection == self.ConnectionDirection.RIGHT) or (lastDirection == self.ConnectionDirection.LEFT and thisDirection == self.ConnectionDirection.DOWN):
            return self.CornerType.TOP_LEFT
        elif (lastDirection == self.ConnectionDirection.RIGHT and thisDirection == self.ConnectionDirection.DOWN) or (lastDirection == self.ConnectionDirection.UP and thisDirection == self.ConnectionDirection.LEFT):
            return self.CornerType.TOP_RIGHT
        elif (lastDirection == self.ConnectionDirection.DOWN and thisDirection == self.ConnectionDirection.LEFT) or (lastDirection == self.ConnectionDirection.RIGHT and thisDirection == self.ConnectionDirection.UP):
            return self.CornerType.BOTTOM_RIGHT
        elif (lastDirection == self.ConnectionDirection.LEFT and thisDirection == self.ConnectionDirection.UP) or (lastDirection == self.ConnectionDirection.DOWN and thisDirection == self.ConnectionDirection.RIGHT):
            return self.CornerType.BOTTOM_LEFT
            
        return self.CornerType.UNDEFINED

    def drawCorner(self, painter, position, cornerType, maxRadius=None):
        #logging.debug(self.__class__.__name__ +": drawCorner() "+ self.cornerTypeString(cornerType))
        thickness = self.CONNECTION_THICKNESS * self.zoomFactor()
        halfthick = thickness / 2
        cornerRoundness = halfthick ** 0.5
        cornerOffset = halfthick * (cornerRoundness)
        innerCorner = halfthick * (cornerRoundness - 1)
        outerCorner = halfthick * (cornerRoundness + 1)
        innerWidth = halfthick * (cornerRoundness - 1)
        radius = halfthick * (cornerRoundness + 1)
        if maxRadius:
            maxRadius = max(maxRadius, thickness)
            radius = min(radius, maxRadius)

        if cornerType == self.CornerType.TOP_RIGHT:
            startAngle = 0

            outerCorner = QPointF(position.x() + halfthick - 2 * radius, position.y() - halfthick)
            innerCorner = QPointF(outerCorner.x(), outerCorner.y() + (thickness))
            center = QPointF(outerCorner.x() + radius, outerCorner.y() + radius)
            
            outerRect = QRectF(outerCorner, QSizeF(2 * radius, 2 * radius))
            innerRect = QRectF(innerCorner, QSizeF((2 * radius - thickness), (2 * radius - thickness)))
            
            outerStart = QPointF(outerCorner.x() + 2 * radius, outerCorner.y() + (radius + halfthick))
            innerStart = QPointF(outerCorner.x() + (radius - halfthick), outerCorner.y())
            
        elif cornerType == self.CornerType.TOP_LEFT:
            startAngle = 90
            
            outerCorner = QPointF(position.x() - halfthick, position.y() - halfthick)
            innerCorner = QPointF(outerCorner.x() + (thickness), outerCorner.y() + (thickness))
            center = QPointF(outerCorner.x() + radius, outerCorner.y() + radius)
            
            outerRect = QRectF(outerCorner, QSizeF(2 * radius, 2 * radius))
            innerRect = QRectF(innerCorner, QSizeF((2 * radius - thickness), (2 * radius - thickness)))
            
            outerStart = QPointF(outerCorner.x() + (radius + halfthick), outerCorner.y())
            innerStart = QPointF(outerCorner.x(), outerCorner.y() + (radius + halfthick))
            
        elif cornerType == self.CornerType.BOTTOM_LEFT:
            startAngle = 180
            
            outerCorner = QPointF(position.x() - halfthick, position.y() + halfthick - 2 * radius)
            innerCorner = QPointF(outerCorner.x() + (thickness), outerCorner.y())
            center = QPointF(outerCorner.x() + radius, outerCorner.y() + radius)
            
            outerRect = QRectF(outerCorner, QSizeF(2 * radius, 2 * radius))
            innerRect = QRectF(innerCorner, QSizeF((2 * radius - thickness), (2 * radius - thickness)))
            
            outerStart = QPointF(outerCorner.x(), outerCorner.y() + (radius - halfthick))
            innerStart = QPointF(outerCorner.x() + (radius + halfthick), outerCorner.y() + (2 * radius))
            
        elif cornerType == self.CornerType.BOTTOM_RIGHT:
            startAngle = 270
            
            outerCorner = QPointF(position.x() + halfthick - 2 * radius, position.y() + halfthick - 2 * radius)
            innerCorner = QPointF(outerCorner.x(), outerCorner.y())
            center = QPointF(outerCorner.x() + radius, outerCorner.y() + radius)
            
            outerRect = QRectF(outerCorner, QSizeF(2 * radius, 2 * radius))
            innerRect = QRectF(innerCorner, QSizeF((2 * radius - thickness), (2 * radius - thickness)))
            
            outerStart = QPointF(outerCorner.x() + (radius - halfthick), outerCorner.y() + 2 * radius)
            innerStart = QPointF(outerCorner.x() + 2 * radius, outerCorner.y() + (radius - halfthick))
            
        else:
            # No defined corner, so nothing to draw.
            #print "PointToPointConnection.drawCorner() - No valid corner, aborting..."
            return
        
        if painter.redirected(painter.device()):
            # e.q. QPixmap.grabWidget()
            painter.setBrush(self.FILL_COLOR1)
        else:
            brush = QRadialGradient(center, radius)
            if radius >= thickness:
                brush.setColorAt((radius - thickness) / radius, self.FILL_COLOR1)   # inner border 
                brush.setColorAt((radius - halfthick + 1) / radius, self.FILL_COLOR2)   # center of line
            else:
                # If zoom is too small use single color
                brush.setColorAt(0, self.FILL_COLOR1)    
            brush.setColorAt(1, self.FILL_COLOR1)                                   # outer border
            painter.setBrush(brush)
        
        path = QPainterPath()
        path.moveTo(outerStart)
        path.arcTo(outerRect, startAngle, 90)
        path.lineTo(innerStart)
        path.arcTo(innerRect, startAngle + 90, - 90)
        path.closeSubpath()
            
        #painter.setPen(Qt.NoPen)
        painter.drawPath(path)
                        
            
#        # Helper lines
#            
#        painter.setBrush(Qt.NoBrush)
#        painter.setPen(QPen(QColor(0,255,255)))
#        painter.drawPath(path)
#        painter.setPen(QPen(QColor(0,255,0)))
#        painter.drawRect(innerRect)
#        painter.setPen(QPen(QColor(0,0, 255)))
#        painter.drawRect(outerRect)
#            
#        # Mark important points
#        painter.setPen(QPen(QColor(0,0, 0)))
#        painter.drawEllipse(outerCorner, 2, 2)
#        painter.drawEllipse(innerCorner, 2, 2)
#        painter.drawEllipse(center, 2, 2)
#        painter.drawEllipse(outerStart, 2, 2)
#        painter.drawEllipse(innerStart, 2, 2)

    def getRectBetweenTwoPoints(self, firstP, secondP, firstCorner=CornerType.UNDEFINED, secondCorner=CornerType.UNDEFINED):
        
        halfthick = 0.5 * self.CONNECTION_THICKNESS * self.zoomFactor() 
        cornerRoundness = halfthick ** 0.5
        offset = 2*halfthick  #* (cornerRoundness + 1) - 1       # -1 prevents one pixel gaps which sometimes appear at corners.
        
        direction = self.getPointToPointDirection(firstP, secondP)
        
        firstOffset = 0
        if self.cornerIsDefined(firstCorner):
            firstOffset = offset
            
        secondOffset = 0
        if self.cornerIsDefined(secondCorner):
            secondOffset = offset
        
        topLeft = QPointF()
        bottomRight = QPointF()
        if direction == self.ConnectionDirection.LEFT:
            # horizontal, negative
            topLeft.setX(secondP.x() + secondOffset)
            topLeft.setY(secondP.y() - halfthick)
            bottomRight.setX(firstP.x() - firstOffset + 1)
            bottomRight.setY(firstP.y() + halfthick)
        elif direction == self.ConnectionDirection.RIGHT:
            # horizontal, positive
            topLeft.setX(firstP.x() + firstOffset)
            topLeft.setY(firstP.y() - halfthick)
            bottomRight.setX(secondP.x() - secondOffset + 1)
            bottomRight.setY(secondP.y() + halfthick)
        elif direction == self.ConnectionDirection.UP:
            # vrtical, negative
            topLeft.setX(secondP.x() - halfthick)
            topLeft.setY(secondP.y() + secondOffset)
            bottomRight.setX(firstP.x() + halfthick)
            bottomRight.setY(firstP.y() - firstOffset + 1)
        elif direction == self.ConnectionDirection.DOWN:
            # vertical, positive
            topLeft.setX(firstP.x() - halfthick)
            topLeft.setY(firstP.y() + firstOffset)
            bottomRight.setX(secondP.x() + halfthick)
            bottomRight.setY(secondP.y() - secondOffset + 1)
        else:
            return QRectF(topLeft, bottomRight)
        
        return QRectF(topLeft, bottomRight)
    
    def drawLineSection(self, painter, firstP, secondP, firstCorner, secondCorner):
        direction = self.getPointToPointDirection(firstP, secondP)
        
        thickness = self.CONNECTION_THICKNESS * self.zoomFactor()
        halfthick = thickness / 2
        cornerRoundness = halfthick ** 0.5
        cornerOffset = halfthick * cornerRoundness * (4 * self.zoomFactor()**2)
        innerCorner = halfthick * (cornerRoundness + 1)
        outerCorner = halfthick * (cornerRoundness - 1)
        
        rect = self.getRectBetweenTwoPoints(firstP, secondP, firstCorner, secondCorner)
            # Paint witch color gradient (PyQt4)
        if direction == self.ConnectionDirection.LEFT:     # horizontal, negative
            brush = QLinearGradient(rect.x(), rect.y(), rect.x(), rect.y() + halfthick)
        elif direction == self.ConnectionDirection.RIGHT:  # horizontal, positive
            brush = QLinearGradient(rect.x(), rect.y(), rect.x(), rect.y() + halfthick)
        elif direction == self.ConnectionDirection.UP:     # vertical, negative
            brush = QLinearGradient(rect.x(), rect.y(), rect.x() + halfthick, rect.y())
        elif direction == self.ConnectionDirection.DOWN:   # vertical, positive
            brush = QLinearGradient(rect.x(), rect.y(), rect.x() + halfthick, rect.y())
        
        brush.setSpread(QLinearGradient.ReflectSpread)
        brush.setColorAt(0, self.FILL_COLOR1)
        brush.setColorAt(1, self.FILL_COLOR2)
        painter.setBrush(brush)
        
        painter.drawRect(rect)
    
    def drawSection(self, painter, sectionIndex):
        """ This is going to replace drawLineSection
        """
        firstP = self.mapFromParent(self._route[sectionIndex])
        secondP = self.mapFromParent(self._route[sectionIndex +1])
        direction = self.getPointToPointDirection(firstP, secondP)
        
        if self.CONNECTION_TYPE!="ORTHOGONAL" or direction == self.ConnectionDirection.UNDEFINED:
            self.drawStraightLine(painter, firstP, secondP)
            return
        
        previousP = None
        nextP = None
        if sectionIndex == 0:
            lastDirection = self.sourceDirection()
        else:
            previousP = self.mapFromParent(self._route[sectionIndex -1])
            lastDirection = self.getPointToPointDirection(previousP, firstP)
        if sectionIndex > len(self._route) -3:
            nextDirection = self.targetDirection()
        else:
            nextP = self.mapFromParent(self._route[sectionIndex +2])
            nextDirection = self.getPointToPointDirection(secondP, nextP)
        
        firstCorner = self.getCornerType(lastDirection, direction)
        secondCorner = self.getCornerType(direction, nextDirection)
        
        minDist = self.CONNECTION_THICKNESS * self.zoomFactor() * 4
        maxRadius = None
        if previousP:
            xDist = abs(firstP.x() - previousP.x())
            yDist = abs(firstP.y() - previousP.y())
            if xDist > 0 and xDist < minDist:
                maxRadius = 0.5 * xDist
            elif yDist > 0 and yDist < minDist:
                maxRadius = 0.5 * yDist
                
        xDist = abs(firstP.x() - secondP.x())
        yDist = abs(firstP.y() - secondP.y())
        if xDist > 0 and xDist < minDist:
            maxRadius = 0.5 * xDist
        elif yDist > 0 and yDist < minDist:
            maxRadius = 0.5 * yDist
        #if maxRadius:
        self.drawCorner(painter, firstP, firstCorner, maxRadius)
        
#        print "_____________________ darawSection _______________________"
#        print "firstP", firstP
#        print "secondP", secondP
#        print "lastDirection", self.connectionDirectionString(lastDirection)
#        print "  firstCorner", self.cornerTypeString(firstCorner)
#        print "direction", self.connectionDirectionString(direction)
#        print "  secondCorner", self.cornerTypeString(secondCorner)
#        print "nextDirection", self.connectionDirectionString(nextDirection)
#        print "\n\n"
        
        thickness = self.CONNECTION_THICKNESS * self.zoomFactor()
        halfthick = thickness / 2
        cornerRoundness = halfthick ** 0.5
        cornerOffset = halfthick * cornerRoundness * (4 * self.zoomFactor()**2)
        innerCorner = halfthick * (cornerRoundness + 1)
        outerCorner = halfthick * (cornerRoundness - 1)
        
        rect = self.getRectBetweenTwoPoints(firstP, secondP, firstCorner, secondCorner)
            # Paint witch color gradient (PyQt4)
        if direction == self.ConnectionDirection.LEFT:     # horizontal, negative
            brush = QLinearGradient(rect.x(), rect.y(), rect.x(), rect.y() + halfthick)
        elif direction == self.ConnectionDirection.RIGHT:  # horizontal, positive
            brush = QLinearGradient(rect.x(), rect.y(), rect.x(), rect.y() + halfthick)
        elif direction == self.ConnectionDirection.UP:     # vertical, negative
            brush = QLinearGradient(rect.x(), rect.y(), rect.x() + halfthick, rect.y())
        elif direction == self.ConnectionDirection.DOWN:   # vertical, positive
            brush = QLinearGradient(rect.x(), rect.y(), rect.x() + halfthick, rect.y())
        else:
            # Should already be drawn above --> direction == self.ConnectionDirection.UNDEFINED
            return
        
        if painter.redirected(painter.device()):
            # e.q. QPixmap.grabWidget()
            painter.setBrush(self.FILL_COLOR1)
        else:
            brush.setSpread(QLinearGradient.ReflectSpread)
            brush.setColorAt(0, self.FILL_COLOR1)
            brush.setColorAt(1, self.FILL_COLOR2)
            painter.setBrush(brush)
        
        painter.drawRect(rect)
            
    def drawStraightLine(self, painter, firstP, secondP):
        """ Draw a straight line between two points.
        """
        thickness = self.CONNECTION_THICKNESS * self.zoomFactor()
        halfthick = max(0.5, thickness / 2)
        pen = QPen(self.FILL_COLOR2, 2 * halfthick, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(firstP, secondP)

    def updateConnection(self):
        """ Recalculates route and then positions and sizes the widget accordingly.
        """
        #logging.debug(self.__class__.__name__ +": updateConnection()")
        #print "          sourcePoint, targetPoint", self.sourcePoint(), self.targetPoint()
        if self.calculateRoute():
            tl = self.topLeft()
            br = self.bottomRight()
            self.move(tl)
            self.resize(br.x() - tl.x(), br.y() - tl.y())
            self.update()
            return True
        return False
    
    def paintEvent(self, event):
        """ Handles paint events.
        """
        if self.updateConnection():
            event.ignore()
        else:
            #print "paintEvent() accept"
            self.draw()
    
    def draw(self):
        """ Draws connection.
        """
        self.calculateRoute()
        if not self.routeIsValid():
            return
        painter = QPainter(self)
        #logging.debug(self.__class__.__name__ +": draw()")
        
        if self._selectedFlag:
            # Selected
            framePen = QPen(self.SELECT_COLOR)
            framePen.setWidth(self.SELECTED_FRAME_WIDTH)
        else:
            #self.select(False)
            framePen = QPen(Qt.NoPen)
            
        #if hasattr(QPainter, 'Antialiasing'):
        if self.zoom() > 30:
            painter.setRenderHint(QPainter.Antialiasing)            
            
#        painter.setPen(Qt.black)
#        for thisP in self._route:
#            painter.drawEllipse(self.mapFromParent(thisP), self.CONNECTION_THICKNESS * self.zoomFactor(), self.CONNECTION_THICKNESS* self.zoomFactor())
            
        painter.setPen(framePen)
        for i in range(0, len(self._route) -1):
            #self.drawLineSection(painter, route[i], route[i + 1], self._cornerTypes[i], self._cornerTypes[i + 1])
            self.drawSection(painter, i)
            #self.drawCorner(painter, route[i], self._cornerTypes[i])
    
    def topLeft(self):
        """ Places a rect around the route and returns the top-left point in parent's coordinates.
        """
        if not self.routeIsValid():
            return None
        thickness = self.CONNECTION_THICKNESS * self.zoomFactor()
        xMin = self._route[0].x()
        yMin = self._route[0].y()
        for point in self._route:
            if point.x() < xMin:
                xMin = point.x()
            if point.y() < yMin:
                yMin = point.y()

        return QPoint(xMin - thickness, yMin - thickness)
    
    def bottomRight(self):
        """ Places a rectangle around the route and returns the bottom-right point in parent's coordinates.
        """
        if not self.routeIsValid():
            return None
        thickness = self.CONNECTION_THICKNESS * self.zoomFactor()
        xMax = self._route[0].x()
        yMax = self._route[0].y()
        for point in self._route:
            if point.x() > xMax:
                xMax = point.x()
            if point.y() > yMax:
                yMax = point.y()
                
        return QPoint(xMax + thickness, yMax + thickness)
    
    def mousePressEvent(self, event):
        """ Selects connection if event.pos() lies within the connection's borders.
        
        Otherwise the event is propagated to underlying widget via AnalysisDesignerWorkspace.propagateEventUnderConnectionWidget().
        """
        logging.debug(__name__ + ": mousePressEvent")
        if self.belongsToRoute(self.mapToParent(event.pos())):
            self.select()
        else:
            if not hasattr(self.parent(), "propagateEventUnderConnectionWidget") or not self.parent().propagateEventUnderConnectionWidget(self, event):
                event.ignore()
            
    def keyPressEvent(self, event):
        """ Handle delete and backspace keys to delete connections.
        """
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            #print __name__ +'.keyPressEvent', event.key()
            self.delete()
            self.emit(SIGNAL("deleteButtonPressed"))

    def delete(self):
        """ Deletes this connection.
        """
        if self._deletedFlag:
            return False
        if not self.isDeletable():
            return False
        self.deleteLater()
        self.emit(SIGNAL("connectionDeleted"))
        return True

class PortConnection(PointToPointConnection):
    """ Connection line between to PortWidgets.
    """
    def __init__(self, workspace, sourcePort, sinkPort):
        """ Constructor.
        
        Creates connection from source port widget to sink port widget.
        """
        self._sourcePort = sourcePort
        self._sinkPort = sinkPort
        PointToPointConnection.__init__(self, workspace, None, None)
        self._sourcePort.attachConnection(self)
        self._sinkPort.attachConnection(self)
        
    def sourcePort(self):
        """ Returns attached source port.
        """
        return self._sourcePort
    
    def sinkPort(self):
        """ Returns attached sink port.
        """
        return self._sinkPort
    
    def sourcePoint(self):
        """ Returns connection point of attached source port.
        """
        return self._sourcePort.connectionPoint()
    
    def targetPoint(self):
        """ Returns connection point of attached sink port.
        """
        return self._sinkPort.connectionPoint()
    
    def sourceDirection(self):
        """ Returns initial direction of source port.
        """
        return self._sourcePort.connectionDirection()
    
    def targetDirection(self):
        """ Returns initial direction of sink port.
        """
        return self._sinkPort.connectionDirection()
    
    def attachedToPort(self, port):
        """ Returns True if port is either source or sink port attached to this connection.
        """
        if port == self._sourcePort:
            return True
        if port == self._sinkPort:
            return True
        return False
    
    def delete(self):
        if PointToPointConnection.delete(self):
            self._sinkPort.detachConnection(self)
            self._sourcePort.detachConnection(self)

class LinearPortConnection(PortConnection):
    """ PortConnection with linear connection
    """
    CONNECTION_TYPE = "STRAIGHT"