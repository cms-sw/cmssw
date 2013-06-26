#!/bin/sh

xsltproc emuMerge.xsl emuDQMCanvases.xml > .emuDocumentationLayout.xml
xsltproc --xinclude emuCanvases2Layouts.xsl .emuDocumentationLayout.xml
rm .emuDocumentationLayout.xml
