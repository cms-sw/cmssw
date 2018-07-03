#!/usr/bin/env python
# This script creates a .tex-file for displaying the results
# of an offline alignment validation of the CMS tracker.
#
# HOW TO USE:
# -Pass the paths of the directories containing the plots as arguments
#  (these would be the ExtendedValidationImages-directories of the
#  relevant validations). E.g.
#    produceOfflineValidationTex.py plotdir1 plotdir2
# -To change the general properties of the presentation to be produced,
#  modify the templates found in presentationTemplates.py.
# -To produce different plots, use the writePageReg-function to select
#  the types of files you want on a page.
#
#
# Originally by Eemeli Tomberg, 2014



import os
import stat
import sys
import time

from Alignment.OfflineValidation.TkAlAllInOneTool.presentation import *
from Alignment.OfflineValidation.TkAlAllInOneTool.presentationTemplates import *

# Script execution.
def main():
    print 'Producing a .tex file from plots...'

    # Get plots from given paths
    if len(sys.argv) < 2:
        print 'Error: Need path of plots as an argument!'
        sys.exit(1)

    validations = []
    for plotpath in sys.argv[1:]:
        validations.append(ValidationPlots(plotpath))

    classes = validationclasses(validations)

    # Compose .tex frames
    frames = ''
    for cls in classes:
        for subsection in cls.presentationsubsections():
            frames += subsection.write([_ for _ in validations if _.validationclass == cls])
    # Summary
    frames += SummarySection().write(validations)
    # Additional plots
    #frames += writePageReg('YourRegExp', 'PageTitle', validations)

    # Write final .tex file
    file = open('presentation.tex', 'w')
    file.write(texTemplate.replace('[frames]', frames).\
               replace('[time]', time.ctime()))
    file.close()

    # A script to get from .tex to .pdf
    pdfScript = open('toPdf.sh', 'w')
    pdfScript.write(toPdf)
    pdfScript.close()
    os.chmod("toPdf.sh", stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)



if __name__ == '__main__':
    main()
