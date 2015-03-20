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



import sys
import re
import os
import stat
import math
import time
from Alignment.OfflineValidation.TkAlAllInOneTool.presentationTemplates import *

subdetectors = ["BPIX", "FPIX", "TIB", "TOB", "TID", "TEC"]


# Plots related to a single validation:
class ValidationPlots:
    def __init__(self, path):
        if not os.path.isdir(path):
            print "Error: Directory "+path+" not found!"
            exit(1)
        if not path.endswith('/'):
            path += '/'
        path = path.replace('\\', '/') # Beacause LaTeX has issues with '\'.
        self.path = path
        # List of plot files in given directory:
        self.plots = [file for file in os.listdir(path)
                      if file.endswith('.eps')]


# Layout of plots on a page:
class PageLayout:
    def __init__(self, pattern=[], width=1, height=1):
        self.pattern = [] # List of rows; row contains the order numbers
                          # of its plots; e.g. [[1,2,3], [4,5,6]]
        self.width = width # Maximum width of one plot,
                           # with respect to textwidth.
        self.height = height # Maximum height of one plot,
                             # with respect to textheight.

    # Sets variables for the given plots and returns the plots
    # in an appropriate order:
    def fit(self, plots):
        rowlengths = []
        # First, try to place plots in a square.
        nplots = sum(len(p) for p in plots)
        length = int(math.ceil(math.sqrt(nplots)))
        # Then, fill the square from the bottom and remove extra rows.
        fullRows = int(nplots/length)
        residual = nplots - length*fullRows
        nrows = fullRows
        if residual != 0:
            rowlengths.append(residual)
            nrows += 1
        for _ in xrange(fullRows):
            rowlengths.append(length)

        # Now, fill the pattern.
        self.pattern = []
        if residual == 0 and len(plots[0])%length != 0 and\
           len(plots[0])%nrows == 0:
            # It's better to arrange plots in columns, not rows.
            self.pattern.extend(range(i, i+nrows*(length-1)+1, nrows)
                                for i in range(1, nrows+1))
        else:
            if residual != 0:
                self.pattern.append(range(1, 1+residual))
            self.pattern.extend(range(i, i+length) for i in
                                range(residual+1, nplots-length+2, length))

        self.width = 1.0/length
        self.height = 0.8/nrows


# Write a set of pages, one for each subdetector.
# Arguments: identifier: regular expression to get the wanted plots,
#                        used together with subdetector name
#            title: title of the plot type
#            validations: list of relevant ValidationPlots objects.
# Returns the parsed script.
def writeSubsection(identifier, title, validations):
    script = ''
    for subdetector in subdetectors:
        script += writePageReg('(?=.*%s)%s'%(subdetector, identifier),
                               title+': ' +subdetector, validations)
    if script != '':
        script = subsectionTemplate.replace('[title]', title)+script
    return script


# Write a page containing plots of given type.
# Arguments: identifier: regular expression to get the wanted plots
#            title: title of the plot type
#            validations: list of relevant ValidationPlots objects
#            layout: given page layout.
# Returns the parsed script.
def writePageReg(identifier, title, validations, layout=0):
    plots = []
    for validation in validations:
        valiplots = [validation.path+plot for plot in validation.plots
                     if re.search(identifier, plot)]
        valiplots.sort(key=plotSortKey)
        plots.append(valiplots)
    if sum(len(p) for p in plots) == 0:
        print 'Warning: no plots matching ' + identifier
        return ''

    # Create layout, if not given.
    if layout == 0:
        layout = PageLayout()
        layout.fit(plots)

    return writePage([p for vali in plots for p in vali], title, layout)


# Write the given plots on a page.
# Arguments: plots: paths of plots to be drawn on the page
#            title: title of the plot type
#            layout: a PageLayout object definig the layout.
# Returns the parsed script.
def writePage(plots, title, layout):
    plotrows = []
    for row in layout.pattern:
        plotrow = []
        for i in xrange(len(row)):
            plotrow.append(plotTemplate.replace('[width]', str(layout.width)).\
                           replace('[height]', str(layout.height)).\
                           replace('[path]', plots[row[i]-1]))
        plotrows.append('\n'.join(plotrow))
    script = ' \\\\\n'.join(plotrows)

    return frameTemplate.replace('[plots]', script).replace('[title]', title)


# Sort key to rearrange a plot list.
# Arguments: plot: to be sorted.
def plotSortKey(plot):
    # Move normchi2 before chi2Prob
    if plot.find('normchi2') != -1:
        return 'chi2a'
    if plot.find('chi2Prob') != -1:
        return 'chi2b'
    return plot






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

    # Compose .tex frames
    frames = ''
    # chi2
    frames += subsectionTemplate.replace('[title]', 'Chi^2 plots')
    frames += writePageReg('chi2', r'$\chi^2$ plots', validations)
    # DMRs
    frames += writeSubsection('DmedianY*R.*plain.eps$', 'DMR', validations)
    # Split DMRs
    frames += writeSubsection('DmedianY*R.*split.eps$','Split DMR',validations)
    # DRnRs:
    frames += writeSubsection('DrmsNY*R.*plain.eps$', 'DRnR', validations)
    # Surface Shapes
    frames += writeSubsection('SurfaceShape', 'Surface Shape', validations)
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
