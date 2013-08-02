import sys
from ge11Dimensions import *

def writeHeader():
    """Print the XML file header"""
    print '<?xml version="1.0"?>'
    print '<DDDefinition xmlns="http://www.cern.ch/cms/DDL" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.cern.ch/cms/DDL ../../../../DetectorDescription/Schema/DDLSchema.xsd">'


def writeConstantsSection():
    """Print the section containing chamber dimensions"""
    
    print 
    print '<ConstantsSection label="gemf.xml" eval="true">'
    print '  <Constant name="dzTot" value="%f*%s"/>'%(dzTot,dzTotDim)
    print '  <Constant name="rPos"  value="(%f*%s+[dzTot])"/>'%(dBeamLine,dBeamLineDim)
    for i in range(1,nEta+1):
        print '  <Constant name="dzS%d"  value="%f*%s"/>'%(i,dzEta[i-1],etaDim)

    ## print gap 
    print '  <Constant name="dzGap" value="%f*%s"/>'%(dzGap,dzGapDim)

    print '  <Constant name="dzIn"  value="(',
    for i in range(1,nEta+1):
        sys.stdout.write("[dzS%d]+"%(i))

    print '%d*[dzGap])"/>'%(nGap)

    print '  <Constant name="z10"   value="([dzIn]-[dzS1])"/> '
    for i in range(2,nEta+1):
        print '  <Constant name="z%d0"   value="([z%d0]-[dzS%d]-2*[dzGap]-[dzS%d])"/>'%(i,i-1,i-1,i)

    print '  <Constant name="dxBot" value="%f*%s"/>'%(dxBot,dxBotDim)
    print '  <Constant name="dxTop" value="%f*%s"/>'%(dxTop,dxTopDim)
    print '  <Constant name="slope" value="([dxTop]-[dxBot])/(2*[dzIn])"/>'

    ## assignment of the z values
    z = [0]*nEta
    z[0] = dzIn - dzEta[0]
    for i in range(1,nEta):
        z[i] = z[i-1] - dzEta[i-1] - 2*dzGap - dzEta[i]

    a=     sum(dzEta) + nGap * dzGap
    print dzTot, dzIn, a
    
    ## assignment of the dx values
    for i in range(1,nEta+1):
        for j in range(1,3):
            if j%2:
                sign = "+"
            else:
                sign = "-"
            
            print '  <Constant name="dx%s%s"  value="([dxTop]-[slope]*([dzIn]-[z%s0]%s[dzS%i]))"'%(i,j,i,sign,i)

    print '</ConstantsSection>'


def writeMaterialSection():
    """Write the materials section"""
    
    f = open("gemMaterialSection.xml",'r')
    print f.read()
    f.close()

def writeSolidSection():
    """Write the solid component section"""

    print '<SolidSection label="gemf.xml">'
    print '  <Tubs name="GE11" rMin="%f*%s  " rMax="%f*%s  " dz="%f*%s " startPhi="0*deg" deltaPhi="360*deg"/>'%(rMin,rMinDim,rMax,rMaxDim, dz, dzDim)
    print """  <Trd1 name="GA11" dz="52.850*cm" dy1="4.500*mm" dy2="4.500*mm" dx1=" 9.411*cm" dx2="15.573*cm" />
  <Trd1 name="GAAX" dz="52.850*cm" dy1="4.000*mm" dy2="4.000*mm" dx1=" 8.910*cm" dx2="15.072*cm" />
  <Trd1 name="GB11" dz="52.850*cm" dy1="7.600*mm" dy2="7.600*mm" dx1="17.227*cm" dx2="26.483*cm" />
  <Trd1 name="GTAX" dz="52.850*cm" dy1="0.500*mm" dy2="0.500*mm" dx1=" 8.910*cm" dx2="15.072*cm" />
  <Trd1 name="GMAX" dz="51.976*cm" dy1="5.500*mm" dy2="5.500*mm" dx1="16.312*cm" dx2="25.390*cm" />
  <Trd1 name="GRAX" dz="50.182*cm" dy1="1.500*mm" dy2="1.500*mm" dx1="13.569*cm" dx2="22.350*cm" />
  <Trd1 name="GIAX" dz="50.182*cm" dy1="1.050*mm" dy2="1.050*mm" dx1="13.569*cm" dx2="22.350*cm" />
  <Trd1 name="GJAX" dz="50.182*cm" dy1="0.500*mm" dy2="0.500*mm" dx1="13.569*cm" dx2="22.350*cm" />
  <Trd1 name="GKAX" dz="48.482*cm" dy1="0.025*mm" dy2="0.025*mm" dx1="12.514*cm" dx2="20.997*cm" />
  <Trd1 name="GSAX" dz="50.182*cm" dy1="1.600*mm" dy2="1.600*mm" dx1="13.569*cm" dx2="22.350*cm" />
  <Trd1 name="GDAX" dz="48.882*cm" dy1="0.125*mm" dy2="0.125*mm" dx1="12.880*cm" dx2="21.433*cm" />"""
    print 
    
    ## gem gas
    gas_layers = ['F','G','H']
    for g in range(len(gas_layers)):
        gas = gas_layers[g]
        ga_dy1 = ga_dy[g]
        for i in range(1,nEta+1):
            print '  <Trd1 name="G%sA%d" dz="[dzS%d]" dy1="%f*%s" dy2="%f*%s" dx1="[dx%d1]" dx2="[dx%d2]" />'%(
                gas,i,i,ga_dy1,ga_dy_dim,ga_dy1,ga_dy_dim,i,i)
        print

    
    ## spacers    
    spacers = ['C','D','E']
    for sp in range(len(spacers)):
        spac = spacers[sp]
        ga_dy1 = ga_dy[sp]
        for i in range(1,nEta+1):
            print '  <Box  name="G%sA%d" dz="[dzS%d]" dy ="%f*%s" dx ="%f*%s" />'%(
                spac,i,i,ga_dy1,ga_dy_dim,ga_dx,ga_dx_dim)
        print 

    print '</SolidSection>'

def writeLogicalPartSection():
    """Write the logical part section"""
    print """<LogicalPartSection label="gemf.xml">
  <LogicalPart name="GE11P" category="unspecified">
    <rSolid name="GE11"/>
    <rMaterial name="materials:ME_free_space"/>
  </LogicalPart>
  <LogicalPart name="GE11N" category="unspecified">
    <rSolid name="GE11"/>
    <rMaterial name="materials:ME_free_space"/>
  </LogicalPart>
  <LogicalPart name="GA11" category="unspecified">
    <rSolid name="GA11"/>
    <rMaterial name="materials:Aluminium"/>
  </LogicalPart>
  <LogicalPart name="GAAX" category="unspecified">
    <rSolid name="GAAX"/>
    <rMaterial name="materials:Air"/>
  </LogicalPart>
  <LogicalPart name="GB11" category="unspecified">
    <rSolid name="GB11"/>
    <rMaterial name="materials:Aluminium"/>
  </LogicalPart>
  <LogicalPart name="GTAX" category="unspecified">
    <rSolid name="GTAX"/>
    <rMaterial name="materials:Air"/>
  </LogicalPart>
  <LogicalPart name="GMAX" category="unspecified">
    <rSolid name="GMAX"/>
    <rMaterial name="materials:Air"/>
  </LogicalPart>
  <LogicalPart name="GRAX" category="unspecified">
    <rSolid name="GRAX"/>
    <rMaterial name="gemf:M_Rdout_Brd"/>
  </LogicalPart>
  <LogicalPart name="GIAX" category="unspecified">
    <rSolid name="GIAX"/>
    <rMaterial name="materials:G10"/>
  </LogicalPart>
  <LogicalPart name="GJAX" category="unspecified">
    <rSolid name="GJAX"/>
    <rMaterial name="materials:G10"/>
  </LogicalPart>
  <LogicalPart name="GKAX" category="unspecified">
    <rSolid name="GKAX"/>
    <rMaterial name="gemf:M_GEM_Foil"/>
  </LogicalPart>
  <LogicalPart name="GSAX" category="unspecified">
    <rSolid name="GSAX"/>
    <rMaterial name="materials:G10"/>
  </LogicalPart>
  <LogicalPart name="GDAX" category="unspecified">
    <rSolid name="GDAX"/>
    <rMaterial name="gemf:M_Kapton_Cu"/>
  </LogicalPart>

"""
    ## gem gas
    gas_layers = ['F','G','H']
    for g in range(len(gas_layers)):
        gas = gas_layers[g]
        for i in range(1,nEta+1):
            print """  <LogicalPart name="G%sA%d" category="unspecified">
    <rSolid name="G%sA%d"/>
    <rMaterial name="gemf:M_GEM_Gas"/>
  </LogicalPart>"""%(gas,i,gas,i)

    ## spacers
    spacers = ['D','E','E']
    for sp in spacers:
        for i in range(1,nEta+1):
            print """  <LogicalPart name="G%sA%d" category="unspecified">
    <rSolid name="G%sA%d"/>
    <rMaterial name="materials:G10"/>
  </LogicalPart>"""%(sp,i,sp,i)

    print '</LogicalPartSection>'

def writePosPartSection():
    """Write the positional part section"""
    
    f = open("gemPosPartSection.xml",'r')
    print f.read()
    f.close()

def writeAlgorithmSection():
    """Write the algorithm section"""
    
    f = open("gemAlgorithmSection.xml",'r')
    print f.read()
    f.close()
    

def gemXMLproducer():
    """Produce the entire XML file"""

    writeHeader()
    writeConstantsSection()
    writeMaterialSection()
    writeSolidSection()
    writeLogicalPartSection()
    writePosPartSection()
    writeAlgorithmSection()

if __name__ == "__main__":  
    gemXMLproducer()
