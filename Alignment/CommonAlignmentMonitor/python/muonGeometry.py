class V:  # a boring old vector class
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
  def __add__(self, other): return V(self.x + other.x, self.y + other.y, self.z + other.z)
  def __sub__(self, other): return V(self.x + other.x, self.y + other.y, self.z + other.z)
  def __neg__(self): return V(-self.x, -self.y, -self.z)
  def __repr__(self): return "V(%g, %g, %g)" % (self.x, self.y, self.z)
  def TVector3(self): return "TVector3(%g, %g, %g)" % (self.x, self.y, self.z)

class Disk:  # or Wheel, but I had to choose one name
  def __repr__(self):
    if self.barrel:
      return "<Disk %d endcap=%d wheel=%d>" % (self.rawid, self.endcap, self.wheel)
    else:
      return "<Disk %d endcap=%d disk=%d>" % (self.rawid, self.endcap, self.disk)

  def __init__(self, rawid, endcap):
    self.rawid = rawid
    self.endcap = endcap
    self.barrel = (endcap == 0)

  def setDT(self, wheel):
    self.wheel = wheel
    return self

  def setCSC(self, disk):
    self.disk = disk
    return self
  
  def setLoc(self, *args):
    self.loc = V(*args)
    return self

  def setXhat(self, *args):
    self.xhat = V(*args)
    return self

  def setYhat(self, *args):
    self.yhat = V(*args)
    return self

  def setZhat(self, *args):
    self.zhat = V(*args)
    return self
  
class Chamber(Disk):
  def __repr__(self):
    if self.barrel:
      return "<Chamber %d endcap=%d wheel=%d station=%d sector=%d>" % (self.rawid, self.endcap, self.wheel, self.station, self.sector)
    else:
      return "<Chamber %d endcap=%d disk=%d ring=%d chamber=%d>" % (self.rawid, self.endcap, self.disk, self.ring, self.chamber)

  def setDT(self, wheel, station, sector):
    self.wheel = wheel
    self.station = station
    self.sector = sector
    return self

  def setCSC(self, disk, ring, chamber):
    self.disk = disk
    self.ring = ring
    self.chamber = chamber
    return self

class Layer(Chamber):
  def __repr__(self):
    if self.barrel:
      return "<Layer %d endcap=%d wheel=%d station=%d sector=%d superlayer=%d>" % (self.rawid, self.endcap, self.wheel, self.station, self.sector, self.superlayer)
    else:
      return "<Layer %d endcap=%d disk=%d ring=%d chamber=%d layer=%d>" % (self.rawid, self.endcap, self.disk, self.ring, self.chamber, self.layer)

  def setDT(self, wheel, station, sector, superlayer):
    self.wheel = wheel
    self.station = station
    self.sector = sector
    self.superlayer = superlayer
    return self
    
  def setCSC(self, disk, ring, chamber, layer):
    self.disk = disk
    self.ring = ring
    self.chamber = chamber
    self.layer = layer
    return self

##########################################################

disks = {}
chambers = {}
layers = {}

import os as _os, sys as _sys
for path in _sys.path:
  geometryDataPath = "%s/Alignment/CommonAlignmentMonitor/muonGeometryData.py" % path
  try:
    _os.stat(geometryDataPath)
    execfile(geometryDataPath)
    break
  except OSError:
    pass

for _d in disks.values(): _d.Chamber = []
for _c in chambers.values(): _c.Layer = []

# Connect them all up in a meaningful way: Layers to Chambers

for _l in layers.values():
  for _c in chambers.values():
    if _l.barrel and _c.endcap == _l.endcap and _c.wheel == _l.wheel and _c.station == _l.station and _c.sector == _l.sector:
      _l.Chamber = _c
      _c.Layer.append(_l)
      break
    elif not _l.barrel and _c.endcap == _l.endcap and _c.disk == _l.disk and _c.ring == _l.ring and _c.chamber == _l.chamber:
      _l.Chamber = _c
      _c.Layer.append(_l)
      break

# Connect them all up in a meaningful way: Chambers to Disks

for _c in chambers.values():
  for _d in disks.values():
    if _c.barrel and _d.endcap == _c.endcap and _d.wheel == _c.wheel:
      _c.Disk = _d
      _d.Chamber.append(_c)
      break
    elif not _c.barrel and _d.endcap == _c.endcap and _d.disk == _c.disk:
      _c.Disk = _d
      _d.Chamber.append(_c)
      break

for _d in disks.values(): _d.Chamber.sort(lambda a, b: cmp(a.rawid, b.rawid))
for _c in chambers.values(): _c.Layer.sort(lambda a, b: cmp(a.rawid, b.rawid))
