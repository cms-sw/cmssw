#!/bin/sh

xsltproc emuMerge.xsl emuDQMCanvases.xml > .emuDocumentationLayout.xml
xsltproc --xinclude emuDocumentation.xsl .emuDocumentationLayout.xml
rm .emuDocumentationLayout.xml
