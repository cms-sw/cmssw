<?xml version="1.0" encoding="ISO-8859-1"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
  xmlns:fn="http://www.w3.org/2005/02/xpath-functions">

<xsl:output method="xml" omit-xml-declaration="no" indent="yes"/>

<xsl:template match="/">
  <DocLayout>
    <xsl:copy-of select="*"/>
    <include xmlns="http://www.w3.org/2001/XInclude" href="emuDQMBooking.xml"/>
    <include xmlns="http://www.w3.org/2001/XInclude" href="emuDQMBookingAdds.xml"/>
  </DocLayout>
</xsl:template>

</xsl:stylesheet>
