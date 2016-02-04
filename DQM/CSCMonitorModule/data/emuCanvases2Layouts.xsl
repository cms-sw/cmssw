<?xml version="1.0" encoding="ISO-8859-1"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions">

<xsl:output method="text" omit-xml-declaration="yes" indent="no"/>
<!-- xsl:strip-space elements="*"/-->

<xsl:template match="/">
  <xsl:text>def csclayout(i, p, *rows): i["CSC/Layouts/" + p] = DQMItem(layout=rows)
  
</xsl:text>
  <xsl:apply-templates select="//Canvas[Prefix='TOP']" mode="default"/>
  <xsl:apply-templates select="//Canvas[Prefix='EMU']" mode="default"/>
  <!--xsl:apply-templates select="Canvases/Canvas[Prefix='DDU']" mode="default"/-->
  <!--xsl:apply-templates select="Canvases/Canvas[Prefix='CSC']" mode="default"/-->
</xsl:template>

<xsl:template match="Canvas[Prefix='TOP']" mode="default">
  <xsl:apply-templates select="." mode="printme">
    <xsl:with-param name="hprefix"/>
    <xsl:with-param name="tprefix"/>
  </xsl:apply-templates>
</xsl:template>

<xsl:template match="Canvas[Prefix='EMU']" mode="default">
  <xsl:apply-templates select="." mode="printme">
    <xsl:with-param name="hprefix" select="'CSC/Summary/'"/>
    <xsl:with-param name="tprefix" select="'EMU Summary/'"/>
  </xsl:apply-templates>
</xsl:template>

<xsl:template match="Canvas[Prefix='DDU']" mode="default">
  <xsl:param name="id" select="36"/>

  <xsl:if test="$id > 0">
    <xsl:apply-templates select="." mode="printme">
      <xsl:with-param name="hprefix"><xsl:text>CSC/DDUs/DDU_</xsl:text><xsl:number format="01" value="$id"/><xsl:text>/</xsl:text></xsl:with-param>
      <xsl:with-param name="tprefix"><xsl:text>EMU DDUs/DDU_</xsl:text><xsl:number format="01" value="$id"/><xsl:text>/</xsl:text></xsl:with-param>
    </xsl:apply-templates>
    <xsl:apply-templates select="." mode="default">
      <xsl:with-param name="id" select="$id - 1"/>
    </xsl:apply-templates>
  </xsl:if>

</xsl:template>

<xsl:template match="Canvas[Prefix='CSC']" mode="default">
  <xsl:param name="uid" select="24"/>
  <xsl:param name="lid" select="10"/>

  <xsl:variable name="label">
    <xsl:number format="001" value="$uid"/>
    <xsl:text>_</xsl:text>
    <xsl:number format="01" value="$lid"/>
  </xsl:variable>

  <xsl:if test="$uid > 0">
    <xsl:apply-templates select="." mode="printme">
      <xsl:with-param name="hprefix">
        <xsl:text>CSC/CSCs/CSC_</xsl:text>
        <xsl:value-of select="$label"/>
        <xsl:text>/</xsl:text>
      </xsl:with-param>
      <xsl:with-param name="tprefix">
        <xsl:text>EMU CSCs/CSC_</xsl:text>
        <xsl:value-of select="$label"/>
        <xsl:text>/</xsl:text>
      </xsl:with-param>
    </xsl:apply-templates>
    <xsl:apply-templates select="." mode="default">
      <xsl:with-param name="uid">
        <xsl:choose>
          <xsl:when test="$lid = 1"><xsl:value-of select="$uid - 1"/></xsl:when>
          <xsl:otherwise><xsl:value-of select="$uid"/></xsl:otherwise>
        </xsl:choose>
      </xsl:with-param>
      <xsl:with-param name="lid">
        <xsl:choose>
          <xsl:when test="$lid = 1"><xsl:value-of select="10"/></xsl:when>
          <xsl:otherwise><xsl:value-of select="$lid - 1"/></xsl:otherwise>
        </xsl:choose>
      </xsl:with-param>
    </xsl:apply-templates>
  </xsl:if>
</xsl:template>

<xsl:template match="Canvas" mode="printme">
  <xsl:param name="hprefix"/>
  <xsl:param name="tprefix"/>

  <xsl:variable name="display"><xsl:choose><xsl:when test="DisplayInWeb=0">0</xsl:when><xsl:otherwise>1</xsl:otherwise></xsl:choose></xsl:variable>
  <xsl:variable name="padsx" select="NumPadsX"/>
  <xsl:variable name="padsy" select="NumPadsY"/>

  <xsl:if test="$display=1">
    <xsl:text>csclayout(dqmitems,"</xsl:text>
    <xsl:value-of select="$tprefix"/><xsl:value-of select="Title"/>
    <xsl:text>",
  </xsl:text>
    <xsl:variable name="count"><xsl:value-of select="count(./*[substring(name(),1,3) = 'Pad' and number(substring(name(),4))])"/></xsl:variable>
    <xsl:for-each select="./*[substring(name(),1,3) = 'Pad' and number(substring(name(),4))]">
      <xsl:variable name="name" select="."/>
      <xsl:text>	</xsl:text>
      <xsl:choose>
        <xsl:when test="((position() - 1) mod $padsx) = 0">
          <xsl:text>[</xsl:text>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text> </xsl:text>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:text>{'path': "</xsl:text>
      <xsl:value-of select="$hprefix"/><xsl:value-of select="."/>
      <xsl:text>", 'description': "</xsl:text>
      <xsl:text>For information please click &lt;a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#</xsl:text>
      <xsl:value-of select="../Name"/>
      <xsl:text>_</xsl:text>
      <xsl:value-of select="$name"/>
      <xsl:text>\"&gt;here&lt;/a&gt;.</xsl:text>
      <xsl:text>"}</xsl:text>
      <xsl:if test="((position()) mod $padsx) = 0 or position() = $count">
        <xsl:text>]</xsl:text>
      </xsl:if>
      <xsl:if test="$count > position()">
        <xsl:text>,</xsl:text>
      </xsl:if>
      <xsl:if test="$count = position()">
        <xsl:text>)</xsl:text>
      </xsl:if>
<xsl:text>
</xsl:text>
    </xsl:for-each>
    <xsl:text>
</xsl:text>
  </xsl:if>

</xsl:template>

</xsl:stylesheet>
