<?xml version="1.0" encoding="ISO-8859-1"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions">

<xsl:output method="text" omit-xml-declaration="yes" indent="no"/>
<xsl:strip-space elements="*"/>

<xsl:template match="/">
  <xsl:text> 
  // ============== Start generated from emuDQMBooking.xml by emuBooking2RenderPlugin.xsl ==================
  
  </xsl:text>

  <xsl:apply-templates select="Booking/Histogram"/>

  <xsl:text> 
  // ============== End generated from emuDQMBooking.xml by emuBooking2RenderPlugin.xsl ==================
  
  </xsl:text>
</xsl:template>

<xsl:template match="Histogram">
  <xsl:text>if(reMatch(".*/</xsl:text>
  <xsl:value-of select="Name"/>
  <xsl:text>$", o.name)) {
  </xsl:text>

  <xsl:apply-templates select="SetLeftMargin"/>
  <xsl:apply-templates select="SetRightMargin"/>
  <xsl:apply-templates select="SetStats"/>
  <xsl:apply-templates select="SetOptStat"/>
  <xsl:apply-templates select="SetGridx[1]"/>
  <xsl:apply-templates select="SetGridy[1]"/>
  <xsl:if test="SetLogx = 1 or SetLogy = 1 or SetLogz = 1">
  <xsl:text>  if(obj->GetMinimum() == obj->GetMaximum()) {
      obj->SetMaximum(obj->GetMinimum() + 0.01);
    }
  </xsl:text>
  </xsl:if>
  <xsl:apply-templates select="SetLogx[1]"/>
  <xsl:apply-templates select="SetLogy[1]"/>
  <xsl:apply-templates select="SetLogz[1]"/>

  <xsl:text>
    return;
  </xsl:text>
  <xsl:text>}

  </xsl:text>
</xsl:template>

<xsl:template match="SetLeftMargin|SetRightMargin">
  <xsl:text>  gPad-></xsl:text>
  <xsl:value-of select="name()"/>
  <xsl:text>(</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>);
  </xsl:text>
</xsl:template>

<xsl:template match="SetRightMargin">
  <xsl:text>  gPad->SetRightMargin(</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>);
  </xsl:text>
</xsl:template>

<xsl:template match="SetStats">
  <xsl:text>  obj->SetStats(</xsl:text>
  <xsl:choose>
    <xsl:when test=". = 1"><xsl:text>true</xsl:text></xsl:when>
    <xsl:when test=". = 0"><xsl:text>false</xsl:text></xsl:when>
  </xsl:choose>
  <xsl:text>);
  </xsl:text>
</xsl:template>

<xsl:template match="SetOptStat">
  <xsl:text>  gStyle->SetOptStat("</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>");
  </xsl:text>
</xsl:template>

<xsl:template match="SetGridx[1]|SetGridy[1]|SetLogx[1]|SetLogy[1]|SetLogz[1]">
  <xsl:text>  gPad-></xsl:text>
  <xsl:value-of select="name()"/>
  <xsl:text>();
  </xsl:text>
</xsl:template>

</xsl:stylesheet>
