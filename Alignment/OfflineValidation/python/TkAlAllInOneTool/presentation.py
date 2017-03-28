import abc
import math
import os
import re

from genericValidation import ValidationForPresentation, ValidationWithPlotsSummary
from helperFunctions import recursivesubclasses
from presentationTemplates import *
from TkAlExceptions import AllInOneError

# Plots related to a single validation:
class ValidationPlots(object):
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

    @property
    def validationclass(self):
        possiblenames = []
        for cls in recursivesubclasses(ValidationForPresentation):
            if cls.__abstractmethods__: continue
            if cls.plotsdirname() == os.path.basename(os.path.realpath(self.path.rstrip("/"))):
                return cls
            possiblenames.append(cls.plotsdirname())
        raise AllInOneError("{} does not match any of the possible folder names:\n{}".format(self.path, ", ".join(possiblenames)))

def validationclasses(validations):
    from collections import OrderedDict
    classes = [validation.validationclass for validation in validations]
    #remove duplicates - http://stackoverflow.com/a/39835527/5228524
    classes = list(OrderedDict.fromkeys(classes))
    return classes

# Layout of plots on a page:
class PageLayout(object):
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
class SubsectionBase(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, title):
        self.title = title
    def write(self, validations):
        script = '\n'.join(_ for _ in self.pages(validations) if _)
        if script != '':
            script = subsectionTemplate.replace('[title]', self.title)+script
        return script
    @abc.abstractmethod
    def pages(self, validations):
        pass

class SubsectionOnePage(SubsectionBase):
    def __init__(self, identifier, title):
        self.identifier = identifier
        super(SubsectionOnePage, self).__init__(title)
    def pages(self, validations):
        return [writePageReg(self.identifier, self.title, validations)]

class SubsectionFromList(SubsectionBase):
    def __init__(self, identifier, title):
        self.identifier = identifier
        super(SubsectionFromList, self).__init__(title)
    def pages(self, validations):
        return [writePageReg('(?=.*%s)%s'%(pageidentifier, self.identifier),
                             self.title+': ' +pagetitle, validations)
                   for pageidentifier, pagetitle in self.pageidentifiers]
    @abc.abstractproperty
    def pageidentifiers(self):
        pass

class SummarySection(SubsectionBase):
    def __init__(self):
        super(SummarySection, self).__init__("Summary")
    def pages(self, validations):
        return [summaryTemplate.replace('[title]', self.title)
                               .replace('[summary]', validation.validationclass.summaryitemsstring(folder=validation.path, latex=True))
                               .replace("tabular", "longtable") for validation in validations
                                                                if issubclass(validation.validationclass, ValidationWithPlotsSummary)]

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

import geometryComparison
import offlineValidation
import trackSplittingValidation
import primaryVertexValidation
import zMuMuValidation
