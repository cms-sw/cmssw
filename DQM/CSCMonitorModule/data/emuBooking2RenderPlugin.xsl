<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0" 
  xmlns:emu="http://www.phys.ufl.edu/cms/emu/dqm"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions">

<xsl:output method="text" omit-xml-declaration="yes" indent="no"/>
<xsl:strip-space elements="*"/>

<xsl:template match="/">

  <xsl:text> 

  // ============== Start generated from emuDQMBooking.xml by emuBooking2RenderPlugin.xsl ==================
  
  </xsl:text>

  <xsl:apply-templates select="/emu:Booking/emu:Histogram" mode="frame" />

  <xsl:text> 
  // ============== End generated from emuDQMBooking.xml by emuBooking2RenderPlugin.xsl ==================
  
  </xsl:text>

</xsl:template>

<xsl:template match="emu:Histogram" mode="frame">

  <xsl:text>if(reMatch(".*/</xsl:text>
  <xsl:value-of select="emu:Name"/>
  <xsl:text>$", o.name)) {
  </xsl:text>

  <xsl:variable name="def" select="@ref" />

  <xsl:if test="$def != ''">
    <xsl:text>  /** Applying definition [</xsl:text>
    <xsl:value-of select="$def"/>
    <xsl:text>] **/
  </xsl:text>
    <xsl:apply-templates select="/emu:Booking/emu:Definition[@id = $def]" mode="definition" />
  </xsl:if>

  <xsl:text>  /** Applying histogram **/
  </xsl:text>
  <xsl:apply-templates select="." mode="definition" />

  <xsl:text>  return;
  }
  </xsl:text>
</xsl:template>

<xsl:template match="emu:Histogram|emu:Definition" mode="definition">

  <xsl:apply-templates select="emu:SetLeftMargin"/>
  <xsl:apply-templates select="emu:SetRightMargin"/>
  <xsl:apply-templates select="emu:SetStats"/>
  <xsl:apply-templates select="emu:SetOptStat"/>
  <xsl:apply-templates select="emu:SetOption"/>
  <xsl:apply-templates select="emu:SetGridx[1]"/>
  <xsl:apply-templates select="emu:SetGridy[1]"/>
  <xsl:if test="emu:SetLogx = 1 or emu:SetLogy = 1 or emu:SetLogz = 1">
  <xsl:text>  if(obj->GetMinimum() == obj->GetMaximum()) {
      obj->SetMaximum(obj->GetMinimum() + 0.01);
    }
  </xsl:text>
  </xsl:if>
  <xsl:apply-templates select="emu:SetLogx[1]"/>
  <xsl:apply-templates select="emu:SetLogy[1]"/>
  <xsl:apply-templates select="emu:SetLogz[1]"/>

</xsl:template>

<xsl:template match="emu:SetLeftMargin|emu:SetRightMargin">
  <xsl:text>  gPad-></xsl:text>
  <xsl:value-of select="name()"/>
  <xsl:text>(</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>);
  </xsl:text>
</xsl:template>

<xsl:template match="emu:SetRightMargin">
  <xsl:text>  gPad->SetRightMargin(</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>);
  </xsl:text>
</xsl:template>

<xsl:template match="emu:SetStats">
  <xsl:text>  obj->SetStats(</xsl:text>
  <xsl:choose>
    <xsl:when test=". = 1"><xsl:text>true</xsl:text></xsl:when>
    <xsl:when test=". = 0"><xsl:text>false</xsl:text></xsl:when>
  </xsl:choose>
  <xsl:text>);
  </xsl:text>
</xsl:template>

<xsl:template match="emu:SetOptStat">
  <xsl:text>  gStyle->SetOptStat("</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>");
  </xsl:text>
</xsl:template>

<xsl:template match="emu:SetOption">
  <xsl:text>  obj->SetOption("</xsl:text>
  <xsl:value-of select="."/>
  <xsl:text>");
  </xsl:text>
</xsl:template>

<xsl:template match="emu:SetGridx[1]|emu:SetGridy[1]|emu:SetLogx[1]|emu:SetLogy[1]|emu:SetLogz[1]">
  <xsl:text>  gPad-></xsl:text>
  <xsl:value-of select="name()"/>
  <xsl:text>();
  </xsl:text>
</xsl:template>

</xsl:stylesheet>
