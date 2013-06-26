<?xml version="1.0" encoding="ISO-8859-1"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions">

<xsl:output method="text" omit-xml-declaration="yes" indent="no"/>
<xsl:strip-space elements="*"/>

<xsl:variable name="imgurl" select="'%ATTACHURL%/'"/>

<xsl:template match="/">
  <xsl:apply-templates select="/DocLayout/Canvases/Canvas[Prefix='TOP']"/>
  <xsl:apply-templates select="/DocLayout/Canvases/Canvas[Prefix='EMU']"/>
</xsl:template>

<xsl:template match="Canvas">

<xsl:text>
---+++++ </xsl:text><xsl:value-of select="Title"/>
<xsl:text>
</xsl:text>
<xsl:value-of select="normalize-space(Descr)"/>
<xsl:text>
</xsl:text>
    
<xsl:text>%TABLE{ sort="off" tableborder="0" cellborder="0"  valign="top" tablewidth="100%" columnwidths="100%,0%" }%
</xsl:text>
    
    <xsl:for-each select="./*[substring(name(),1,3) = 'Pad' and number(substring(name(),4))]">
      <xsl:variable name="histo" select="."/>
      <xsl:apply-templates select="//Histogram[Name=$histo]">
        <xsl:with-param name="position" select="position()"/>
        <xsl:with-param name="canvas_name" select="../Name"/>
      </xsl:apply-templates>
    </xsl:for-each>

</xsl:template>

<xsl:template match="Histogram">
 <xsl:param name="position" value="2"/>
 <xsl:param name="canvas_name" value="canv"/>

<xsl:text>| *</xsl:text>

<xsl:text>&lt;a name="</xsl:text>
<xsl:value-of select="$canvas_name"/>
<xsl:text>_</xsl:text>
<xsl:value-of select="Name"/>
<xsl:text>"&gt;&lt;/a&gt;</xsl:text>

<xsl:value-of select="Title"/>

<xsl:choose>
<xsl:when test="$position = 1">
<xsl:text>*  | &lt;img src="</xsl:text>
<xsl:value-of select="$imgurl"/>
<xsl:value-of select="$canvas_name"/>
<xsl:text>_ref.png" onclick="this.src='</xsl:text>
<xsl:value-of select="$imgurl"/>
<xsl:value-of select="$canvas_name"/>
<xsl:text>.png'" ondblclick="this.src='</xsl:text>
<xsl:value-of select="$imgurl"/>
<xsl:value-of select="$canvas_name"/>
<xsl:text>_ref.png'" /&gt;&lt;br/&gt;(click - enlarge; dblclick - back) |
</xsl:text>
</xsl:when>
<xsl:otherwise>
<xsl:text>*  |^|
</xsl:text>
</xsl:otherwise>
</xsl:choose>

<xsl:text>| </xsl:text>

<xsl:choose>
<xsl:when test="string-length(normalize-space(Descr)) > 0">
<xsl:value-of select="normalize-space(Descr)"/>
</xsl:when>
<xsl:otherwise>
<xsl:text>(No description)</xsl:text>
</xsl:otherwise>
</xsl:choose>

<xsl:text>  |^|
</xsl:text>

</xsl:template>

</xsl:stylesheet>
