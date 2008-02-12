<?xml version="1.0" encoding="utf8" ?>

<!--                     	-->
<!-- ECALDB XML DOC STYLESHEET	-->
<!--				-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output method="xml" indent="yes" encoding="utf-8"/>
<xsl:strip-space elements="*" />

	<xsl:template match="ECAL">

		<!-- initial html stuff...-->
		<html>
		<head>
		<title>XML Documentation for ECAL DB numbering</title>
		<link rel="stylesheet" type="text/css" href="style.css"/>
		</head>

		<body>

		<!--HEADER-->
		<div class = "header">

			<img src = "images/CMSlogo.png" />	
			<div id = "title"> XML Documentation for ECAL DB numbering</div>
 			<a href = "instructionsRead.html" target = "blank"> How to read this page</a>
			<a href = "instructionsWrite.html" target = "blank"> How to modify this page</a>
		
		</div>
		
		<!--END HEADER-->


		<div id = "endcap">
		<xsl:apply-templates select="endcap"/>
		</div>

		</body>
		</html>

	</xsl:template>

	<xsl:template match="endcap">

		<xsl:apply-templates select="numbering"/>

	</xsl:template>

	<xsl:template match="numbering">

		<div class= "numbering">

			<p>
			<xsl:text>Part::</xsl:text>
			<xsl:value-of select="normalize-space(part)" />
			
			</p>
	
			<p>
			<xsl:text>Id Names::</xsl:text>
			<xsl:value-of select="normalize-space(idnames)" />
			
			</p>
	
			<p>
			<xsl:text>Description::</xsl:text>
			<xsl:value-of select="normalize-space(description)" />
			
			</p>
	
			<p>
			<xsl:text>Logic_ids::</xsl:text>
			<xsl:value-of select="normalize-space(logic_ids)" />
			
			</p>
	
			<p>
			<xsl:text>Channel_ids::</xsl:text>
			<xsl:value-of select="normalize-space(channel_ids)" />
			
			</p>
	
			<p>
			<xsl:text>Number of channels::</xsl:text>
			<xsl:value-of select="normalize-space(noOfChannels)" />
			
			</p>

		</div>

	</xsl:template>

</xsl:stylesheet>