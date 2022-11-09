#!/usr/bin/env python3

import ROOT
import os
import sys
from decimal import Decimal

class DMRplotter:
    def __init__(self, args):
        self.args = args
        self.dataFiles = []
        self.dataDirs = []
        self.mcFiles = []
        self.fileBaseName = "OfflineValidationSummary.root" 
        self.outputDir = self.args['outputDir']
        self.cwd = os.getcwd() 
        self.objNameList = []
        self.MCobjects = []
        self.objNameListMC = []
        self.segments = ["BPIX","FPIX","TEC","TID","TIB","TOB"]
        self.varsX = {}
        self.legendOffset = 1.5 
        self.legendTextSize = 0.032 
        self.statBoxTextSize = 0.0365 
        self.segmentTextOffset = {'ymin' : 0.9, 'ymax' : 1.2 }
        self.maxEntriesPerColumn = 5
        
    def __log__(self,log_type="",text=""):
        #########################################################################################################################
        #Logger:
        #  INFO    = Informative text
        #  WARNING = Notify user about unpredictable changes or missing files which do not result in abort
        #  ERROR   = Error in logic results in abort. Can be fixed by user (missing input, settings clash ...)
        #  FATAL   = Fatal error results in abort. Cannot be fixed by user (the way how input is produced has changed or bug ...)
        #########################################################################################################################

        v = int(sys.version_info[0])
        source = "DMRplotter:                "
        text = str(text) 
        if v == 3:
            if "i" in log_type:
                print(source,"[INFO]     ",text)
            elif "n" in log_type:
                print("                  ",text)
            elif "w" in log_type:
                print(source,"[WARNING]  ",text)
            elif "e" in log_type:
                print(source,"[ERROR]    ",text)
            elif "f" in log_type:
                print(source,"[FATAL]    ",text)
            else:
                print(text)

    def _replaceMulti(self, mainString, toBeReplaced, newString):
        #################################
        #Auxiliary function to remove 
        #multiple substrings from string
        #################################

        for elem in toBeReplaced:
            if elem in mainString:
                mainString = mainString.replace(elem, newString)    
        return  mainString

    def _styledTPaveText(self,x1,y1,x2,y2,var):
        ####################################
        #Auxiliary function returning styled
        #plain TPaveText.  
        ####################################
        
        textBox = ROOT.TPaveText(x1,y1,x2,y2)
        textBox.SetFillColor(ROOT.kWhite) 
        if "median" not in var or not self.args['useFit']:
            if self.args['showMeanError'] and self.args['showRMSError']:
                textBox.SetTextSize(self.statBoxTextSize-0.008)
            elif self.args['showMean'] and self.args['showRMS'] and (self.args['showMeanError'] or self.args['showRMSError']):
                textBox.SetTextSize(self.statBoxTextSize-0.005)
            else:
                textBox.SetTextSize(self.statBoxTextSize)
        else:
            if self.args['useFitError']:
                textBox.SetTextSize(self.statBoxTextSize-0.008)
            else:
                textBox.SetTextSize(self.statBoxTextSize-0.005)
        textBox.SetTextFont(42)
        
        return textBox   

    def __createSingleArchitecture__(self):
        ########################################
        #Check if input files exist in input dir
        #and creatte output dir 
        ########################################

        duplicity_check = False 
        if len(self.dataDirs) != 0:
            if self.args['isDMR']:
                #subdirectory
                if not os.path.isdir(self.outputDir):
                    self.__log__("i","Creating subdirectory for single DMRs: "+self.outputDir)
                    os.system("mkdir "+self.outputDir)
                else:
                    self.__log__("i","Results directory "+self.outputDir+" exists.") 

                #per IoV directories/MC part directories
                dirsToMake = [] 
                for dataDir in self.dataDirs:  
                    for root,dirs,files in os.walk(dataDir):
                        for dir in dirs: 
                            if dir.startswith("offline"): dirsToMake.append(self.outputDir+"/"+dir)
                for dir in dirsToMake:
                    if not os.path.isdir(dir):
                        os.system("mkdir "+dir) 
                    else:
                        duplicity_check = True
        else:
            self.__log__("e","No input directory found! No DATA or MC present.")
            sys.exit(0) 
        
        if duplicity_check:
            self.__log__("w","Duplicated file names found. Plots will be overwritten.")

    def __createArchitecture__(self):
        ###########################
        #Check if input file exists 
        #and create output dir 
        ###########################
            
        dataControl = True
        for datafile in self.dataFiles: 
            if not os.path.isfile(datafile):
                dataControl = False 
        for mcfile in self.MCobjects:
            if not os.path.isfile(mcfile):
                dataControl = False

        if dataControl and not (len(self.dataFiles) == 0 and len(self.MCobjects) == 0):
            if not os.path.isdir(self.outputDir): 
                self.__log__("i","Final plots will be stored in: "+self.outputDir)
                os.system("mkdir "+self.outputDir)
            else:
                self.__log__("i","Results directory "+self.outputDir+" exists.")
        else:
            self.__log__("f","Results file NOT found! No DATA or MC present.")
            sys.exit(0) 

    def __defineSingleObjects__(self):
        #######################################################
        #Open each file separately and get object groups
        #######################################################

        objDicts = {'DATA' : [], 'MC' : []}

        #DATA
        for datafile in self.dataFiles:
            if not os.path.isfile(datafile): continue
            fInput  = ROOT.TFile.Open(datafile,'READ')
            keyList = ROOT.gDirectory.GetListOfKeys()
            _id = [ id for id in datafile.split("/") if "offline_" in id ]
            id = "0"
            if len(_id) > 0: 
                id = str(_id[0].split("_")[-1]) 
            objDict = {} 
            objList = []
            objAreIgnored = []
            _objNameList = []
            for key in keyList:
                obj = key.ReadObj()
                if "TH1" in obj.ClassName():
                    objList.append(obj.Clone())
                    objName = obj.GetName()
                    #FIXME
                    skipHist = False
                    for tag in ["layer","disc","plus","minus"]:
                        if tag in objName: skipHist = True
                    if skipHist: continue
                    #END FIXME
                    if objName[-1] != "y":
                        generalObjName = self._replaceMulti(objName, [objName.split("_")[0]+"_","_"+objName.split("_")[-1]], "")
                        if len(self.args['objects']) == 0: #get different object names manually
                            if generalObjName not in _objNameList:
                                _objNameList.append(generalObjName)
                        else: #get different object names from user command input
                            if generalObjName not in self.args['objects']:
                                self.__log__("w","Object \""+generalObjName+"\" found but ignored for plotting!")
                                objAreIgnored.append(generalObjName)
                            else:
                                if generalObjName not in _objNameList:
                                    _objNameList.append(generalObjName)
                    self.objNameList = [ genObjName for genObjName in _objNameList ]

            #now fill objects to the structured dictionary
            for objName in self.objNameList:
                objDict[objName] = []
                for obj in objList:
                    if objName in obj.GetName():
                        segment = ""
                        var = ""
                        if obj.GetName()[-1] == "y":
                            segment = obj.GetName().split("_")[-2]
                            var = obj.GetName().split("_")[0]+"Y"
                        else:
                            segment = obj.GetName().split("_")[-1]
                            var = obj.GetName().split("_")[0]+"X"
                        obj.SetDirectory(0) #important to detach memory allocation
                        objDict[objName].append({ 'hist'    : obj,
                                                  'segment' : segment,
                                                  'var'     : var,
                                                  'id'      : id,
                                                  'type'    : "DATA"    
                                                })
            fInput.Close()
            objDicts['DATA'].append(objDict)

        #ensure plotting order
        if len(self.args['objects']) != 0:
            order = []
            for genObjName in self.objNameList:
                order.append(self.args['objects'].index(genObjName))
            orderedList = [self.objNameList[i] for i in order]
            self.objNameList = orderedList

        if len(self.objNameList) == 0 and len(self.dataFiles) !=0:
            self.__log__("e","Data object names (if specified) must correspond to names in given input file!")
            sys.exit(0)
        else:
            for genObjName in self.objNameList:
                self.__log__("i","Object \""+genObjName+"\" found for plotting.")

        #MC
        for mcFile in self.mcFiles:
            fInputMC  = ROOT.TFile.Open(mcFile,'READ')
            keyListMC = ROOT.gDirectory.GetListOfKeys()
            objListMC = []
            objDictMC = {} 
            generalObjName = ""
            objIsIgnored = False
            for key in keyListMC:
                obj = key.ReadObj()
                if "TH1" in obj.ClassName():
                    objName = obj.GetName()
                    objListMC.append(obj.Clone(objName))
                    #FIXME
                    skipHist = False
                    for tag in ["layer","disc","plus","minus"]:
                        if tag in objName: skipHist = True
                    if skipHist: continue
                    #END FIXME
                    if objName[-1] != "y":
                        generalObjName = self._replaceMulti(objName, [objName.split("_")[0]+"_","_"+objName.split("_")[-1]], "")
                        if len(self.args['objects']) == 0: #get different object names manually
                             if generalObjName not in self.objNameListMC:
                                self.objNameListMC.append(generalObjName)
                        else: #get different object names from user command input
                            if generalObjName not in self.args['objects']:
                                self.__log__("w","Object \""+generalObjName+"\" found but ignored for plotting!")
                                objIsIgnored = True
                            else:
                                if generalObjName not in self.objNameListMC:
                                    self.objNameListMC.append(generalObjName)

            #now fill MC objects to the structured dictionary
            if not objIsIgnored:
                objDictMC[generalObjName] = []
                for obj in objListMC:
                    if generalObjName in obj.GetName():
                        segment = ""
                        var = ""
                        if obj.GetName()[-1] == "y":
                            segment = obj.GetName().split("_")[-2]
                            var = obj.GetName().split("_")[0]+"Y"
                        else:
                            segment = obj.GetName().split("_")[-1]
                            var = obj.GetName().split("_")[0]+"X"
                        obj.SetDirectory(0) #important to detach memory allocation
                        objDictMC[generalObjName].append({ 'hist'    : obj,
                                                     'segment' : segment,
                                                     'var'     : var,
                                                     'type'    : "MC"
                                                  })
            fInputMC.Close()
            objDicts['MC'].append(objDictMC)  

        if len(self.objNameListMC) == 0 and len(self.mcFiles) != 0:
            self.__log__("e","MC object names (if specified) must correspond to names in given input file!")
            sys.exit(0)
        else:
            for genObjName in self.objNameListMC:
                self.__log__("i","Object \""+genObjName+"\" found for plotting.")

        return objDicts

    def __defineObjects__(self):
        #################################################################################
        #Open result file and get information about objects stored inside. In case
        #that input validation objects were not given as an argument, it will retrieve 
        #those names from histogram names. Otherwise it will search for particular object
        #names. Histograms are then stored for each module segment and each object.
        #################################################################################

        objDict = {}
        for datafile in self.dataFiles:
            fInput  = ROOT.TFile.Open(datafile,'READ')
            keyList = ROOT.gDirectory.GetListOfKeys()
            objList = []
            objAreIgnored = []
            _objNameList = []
            for key in keyList:
                obj = key.ReadObj()
                if "TH1" in obj.ClassName(): 
                    objList.append(obj.Clone())
                    objName = obj.GetName()
                    #FIXME if you want to average also subsegment histos
                    skipHist = False
                    for tag in ["layer","disc","plus","minus"]:
                        if tag in objName: skipHist = True 
                    if skipHist: continue
                    #END FIXME
                    if objName[-1] != "y":
                        generalObjName = self._replaceMulti(objName, [objName.split("_")[0]+"_","_"+objName.split("_")[-1]], "") 
                        if len(self.args['objects']) == 0: #get different object names manually
                            if generalObjName not in _objNameList:                          
                                _objNameList.append(generalObjName) 
                        else: #get different object names from user command input
                            if generalObjName not in self.args['objects']: 
                                self.__log__("w","Object \""+generalObjName+"\" found but ignored for plotting!")
                                objAreIgnored.append(generalObjName) 
                            else:
                                if generalObjName not in _objNameList:
                                    _objNameList.append(generalObjName)
            duplicates = [ genObjName for genObjName in _objNameList if genObjName in self.objNameList ]
            for dup in duplicates:
                self.__log__("e","Duplicated object "+str(dup)+" was found! Please rename this object in your input file!")
                sys.exit(0) 
            self.objNameList += [ genObjName for genObjName in _objNameList if genObjName not in self.objNameList ]

            #now fill objects to the structured dictionary
            for objName in _objNameList:
                if objName in objAreIgnored: continue   
                objDict[objName] = []
                for obj in objList:
                    if objName in obj.GetName():
                        segment = ""
                        var = ""
                        if obj.GetName()[-1] == "y":
                            segment = obj.GetName().split("_")[-2]
                            var = obj.GetName().split("_")[0]+"Y"
                        else:
                            segment = obj.GetName().split("_")[-1]
                            var = obj.GetName().split("_")[0]+"X"
                        obj.SetDirectory(0) #important to detach memory allocation
                        objDict[objName].append({ 'hist'    : obj,
                                                  'segment' : segment,
                                                  'var'     : var,
                                                  'type'    : "DATA"
                                                })
            fInput.Close()              

        #ensure plotting order
        '''
        if len(self.args['objects']) != 0:
            order = []
            for genObjName in self.objNameList:
                order.append(self.args['objects'].index(genObjName))
            orderedList = [self.objNameList[i] for i in order]
            self.objNameList = orderedList
        '''

        if len(self.objNameList) == 0 and len(self.dataFiles) !=0:
            self.__log__("e","Data object names (if specified) must correspond to names in given input file!")
            sys.exit(0)
        else:
            for genObjName in self.objNameList:
                self.__log__("i","Object \""+genObjName+"\" found for plotting.")
 
        #add MC objects
        for MCobject in self.MCobjects:
            fInputMC  = ROOT.TFile.Open(MCobject,'READ')
            keyListMC = ROOT.gDirectory.GetListOfKeys()
            objListMC = []
            #generalObjName = "" 
            #objIsIgnored = False
            objAreIgnored = []
            _objNameList = [] 
            for key in keyListMC:
                obj = key.ReadObj()
                if "TH1" in obj.ClassName():
                    objName = obj.GetName()
                    objListMC.append(obj.Clone(objName))
                    #FIXME
                    skipHist = False
                    for tag in ["layer","disc","plus","minus"]:
                        if tag in objName: skipHist = True
                    if skipHist: continue
                    #END FIXME
                    if objName[-1] != "y":
                        generalObjName = self._replaceMulti(objName, [objName.split("_")[0]+"_","_"+objName.split("_")[-1]], "")
                        if len(self.args['objects']) == 0: #get different object names manually
                             if generalObjName not in _objNameList:
                                _objNameList.append(generalObjName)
                        else: #get different object names from user command input
                            if generalObjName not in self.args['objects']:
                                self.__log__("w","Object \""+generalObjName+"\" found but ignored for plotting!")
                                objAreIgnored.append(generalObjName)
                            else:
                                if generalObjName not in _objNameList:
                                    _objNameList.append(generalObjName)
            duplicates = [ genObjName for genObjName in _objNameList if genObjName in self.objNameListMC ]
            for dup in duplicates:
                self.__log__("e","Duplicated object "+str(dup)+" was found! Please rename this object in your input file!")
                sys.exit(0)
            self.objNameListMC += [ genObjName for genObjName in _objNameList if genObjName not in self.objNameListMC ]

            #now fill MC objects to the structured dictionary
            for objName in _objNameList:
                if objName in objAreIgnored: continue
                objDict[objName] = []
                for obj in objListMC:
                    if objName in obj.GetName():
                        segment = ""
                        var = ""
                        if obj.GetName()[-1] == "y":
                            segment = obj.GetName().split("_")[-2]
                            var = obj.GetName().split("_")[0]+"Y"
                        else:
                            segment = obj.GetName().split("_")[-1]
                            var = obj.GetName().split("_")[0]+"X"
                        obj.SetDirectory(0) #important to detach memory allocation
                        objDict[objName].append({    'hist'    : obj,
                                                     'segment' : segment,
                                                     'var'     : var,
                                                     'type'    : "MC"
                                                  })
            fInputMC.Close()

        if len(self.objNameListMC) == 0 and len(self.MCobjects) != 0:
            self.__log__("e","MC object names (if specified) must correspond to names in given input file!")
            sys.exit(0)
        else:
            for genObjName in self.objNameListMC:
                self.__log__("i","Object \""+genObjName+"\" found for plotting.")

        #ensure plotting order
        self.objNameList += self.objNameListMC
        if len(self.args['objects']) != 0:
            order = []
            for genObjName in self.objNameList:
                order.append(self.args['objects'].index(genObjName))
            orderedList = [self.objNameList[i] for i in order]
            self.objNameList = orderedList
        return objDict

    def __fitGauss__(self,hist):
        #######################################################################
        # 1. fits a Gauss function to the inner range of abs(2 rms)
        # 2. repeates the Gauss fit in a 3 sigma range around mean of first fit
        # returns mean and sigma from fit in micrometers   
        #######################################################################
 
        if not hist or hist.GetEntries() < 20: return 0
        self.__log__("i","Fitting histogram: "+hist.GetName())

        xScale = 10000. 
        mean = hist.GetMean(1)*xScale
        sigma = hist.GetRMS(1)*xScale
        funcName = "gaussian_"+hist.GetName()
        func = ROOT.TF1(funcName,"gaus",mean - 2.*sigma,mean + 2.*sigma)
        func.SetLineColor(ROOT.kMagenta)
        func.SetLineStyle(2) 
     
        #first fit
        if int(hist.Fit(func,"QNR")) == 0:
            mean = func.GetParameter(1)
            sigma = func.GetParameter(2)
            func.SetRange(mean - 3.*sigma, mean + 3.*sigma)
            # I: Integral gives more correct results if binning is too wide 
            # L: Likelihood can treat empty bins correctly (if hist not weighted...)
            #second fit
            if int(hist.Fit(func,"Q0ILR")) == 0:  
                return func
            else:
                return 0
        else:
            return 0 

    def __getStat__(self,hist,var):
        #############################################################################
        #Return label to be added to the legend for each object. Label describes 
        #statistical information about particular histogram: 
        #(mean+-meanerror) | (rms+-rmserror) for median and RMS plots
        #or fit parameters (mu and sigma) from gaussian fit +-std. deviation error)
        #############################################################################

        statLabel = ""
        delimeter = ""
        muScale = 1.
        muUnit = ""
        form = "{:.2g}"
        formScie = "{:.1e}"
        if "median" in var:
            muScale = 10000.    
            muUnit  = " #mum"
        if not self.args['useFit'] or "median" not in var:  
            if self.args['showMean'] and self.args['showRMS']:
                delimeter = ", "
            if self.args['showMean']:
                statLabel += "#mu="
                if hist.GetMean(1) >= 0.:
                    statLabel += (" "+form).format(Decimal(str(hist.GetMean(1)*muScale)))
                else:
                    statLabel += form.format(Decimal(str(hist.GetMean(1)*muScale)))   
                if self.args['showMeanError']:
                    statLabel += " #pm "
                    statLabel += formScie.format(Decimal(str(hist.GetMeanError(1)*muScale)))
            statLabel += delimeter
            if self.args['showRMS']:
                statLabel += "rms="
                statLabel += (" "+form).format(Decimal(str(hist.GetRMS(1)*muScale)))
                if self.args['showRMSError']:
                    statLabel += " #pm "
                    statLabel += form.format(Decimal(str(hist.GetRMSError(1)*muScale)))
            statLabel += muUnit
        else:
            fitResults = self.__fitGauss__(hist)
            if not isinstance(fitResults, int): 
                delimeter = ", "
                meanFit = fitResults.GetParameter(1)
                meanFitError = fitResults.GetParError(1)
                sigmaFit = fitResults.GetParameter(2)
                sigmaFitError = fitResults.GetParError(2)
                statLabel += "#mu="
                if meanFit >= 0.:
                    statLabel += (" "+formScie).format(Decimal(str(meanFit)))
                    if self.args['useFitError']:
                        statLabel += " #pm "
                        statLabel += form.format(Decimal(str(meanFitError)))
                else:
                    statLabel += formScie.format(Decimal(str(meanFit)))
                    if self.args['useFitError']:
                        statLabel += " #pm "
                        statLabel += form.format(Decimal(str(meanFitError)))  
                statLabel += delimeter
                statLabel += "#sigma="
                statLabel += (" "+form).format(Decimal(str(sigmaFit)))
                if self.args['useFitError']:
                    statLabel += " #pm "
                    statLabel += form.format(Decimal(str(sigmaFitError))) 
                statLabel += muUnit
                        
        return statLabel
        
    def __setTHStyle__(self,objects):
        ##############################################################
        #Set histogram labels, axis titles, line color, stat bar, etc.
        ##############################################################

        #define DMR-specific properties
        varsX = {'medianX' : "median(x\'_{pred}-x\'_{hit})[#mum]",
                 'medianY' : "median(y\'_{pred}-y\'_{hit})[#mum]",
                 'DrmsNRX' : "RMS((x\'_{pred}-x\'_{hit})/#sigma)",
                 'DrmsNRY' : "RMS((y\'_{pred}-y\'_{hit})/#sigma)"
                }
        self.varsX = varsX
        varsY = {'medianX' : "luminosity-weighted number of modules",
                 'medianY' : "luminosity-weighted number of modules", 
                 'DrmsNRX' : "luminosity-weighted number of modules",
                 'DrmsNRY' : "luminosity-weighted number of modules" 
                }
        limitX = {'min' : 10000, 'max' : 10000} 
 
        #set specific style for DMRs
        for objName,objList in objects.items():
            for obj in objList:
                #axis
                scaleFactor ="" 
                obj['hist'].GetXaxis().SetTitle(varsX[obj['var']])
                obj['hist'].GetXaxis().SetTitleFont(obj['hist'].GetYaxis().GetTitleFont()) 
                obj['hist'].GetYaxis().SetTitleSize(0.038) 
                obj['hist'].GetYaxis().SetTitleOffset(1.7)
                if "median" in obj['var']:
                    scaleFactor ="/"+'{:.2f}'.format((obj['hist'].GetXaxis().GetXmax()*limitX['max']-obj['hist'].GetXaxis().GetXmin()*limitX['min'])/obj['hist'].GetXaxis().GetNbins())+" #mum"
                    minX = obj['hist'].GetXaxis().GetXmin()
                    maxX = obj['hist'].GetXaxis().GetXmax()
                    obj['hist'].GetXaxis().SetLimits(minX*limitX['min'],maxX*limitX['max']) 
                obj['hist'].GetYaxis().SetTitle(varsY[obj['var']]+scaleFactor)

                #main title
                obj['hist'].SetTitle("")

                #line color & style
                if len(self.args['objects']) != 0:
                    if obj['type'] == "MC":
                        obj['hist'].SetLineColor(self.args['colors'][self.args['objects'].index(objName)])  
                        obj['hist'].SetLineStyle(self.args['styles'][self.args['objects'].index(objName)])
                        obj['hist'].SetLineWidth(3) #2
                    elif obj['type'] == "DATA":
                        obj['hist'].SetMarkerColor(self.args['colors'][self.args['objects'].index(objName)])
                        obj['hist'].SetLineColor(self.args['colors'][self.args['objects'].index(objName)])
                        obj['hist'].SetMarkerStyle(self.args['styles'][self.args['objects'].index(objName)])
                        obj['hist'].SetMarkerSize(1.5)  

        #set general style for DMRs
        tStyle = ROOT.TStyle("StyleCMS","Style CMS")

        #zero horizontal error bars
        tStyle.SetErrorX(0)

        #canvas settings
        tStyle.SetCanvasBorderMode(0)
        tStyle.SetCanvasColor(ROOT.kWhite)
        tStyle.SetCanvasDefH(800) #800
        tStyle.SetCanvasDefW(800)
        tStyle.SetCanvasDefX(0)
        tStyle.SetCanvasDefY(0)

        #frame settings
        tStyle.SetFrameBorderMode(0)
        tStyle.SetFrameBorderSize(10)
        tStyle.SetFrameFillColor(ROOT.kBlack)
        tStyle.SetFrameFillStyle(0)
        tStyle.SetFrameLineColor(ROOT.kBlack)
        tStyle.SetFrameLineStyle(0)
        tStyle.SetFrameLineWidth(1)
        tStyle.SetLineWidth(2)

        #pad settings
        tStyle.SetPadBorderMode(0)
        tStyle.SetPadColor(ROOT.kWhite)
        tStyle.SetPadGridX(False)
        tStyle.SetPadGridY(False)
        tStyle.SetGridColor(0)
        tStyle.SetGridStyle(3)
        tStyle.SetGridWidth(1) 

        #margins
        tStyle.SetPadTopMargin(0.08)
        tStyle.SetPadBottomMargin(0.13)
        tStyle.SetPadLeftMargin(0.16)
        tStyle.SetPadRightMargin(0.05)

        #common histogram settings
        tStyle.SetHistLineStyle(0)
        tStyle.SetHistLineWidth(3)
        tStyle.SetMarkerSize(0.8)
        tStyle.SetEndErrorSize(4)
        tStyle.SetHatchesLineWidth(1)

        #stat box
        tStyle.SetOptFile(0) 

        #axis settings
        tStyle.SetAxisColor(1,"XYZ")
        tStyle.SetTickLength(0.03,"XYZ")
        tStyle.SetNdivisions(510,"XYZ")
        tStyle.SetPadTickX(1)
        tStyle.SetPadTickY(1)
        tStyle.SetStripDecimals(ROOT.kFALSE)

        #axis labels and titles
        tStyle.SetTitleColor(1,"XYZ")
        tStyle.SetLabelColor(1,"XYZ")
        tStyle.SetLabelFont(42,"XYZ")
        tStyle.SetLabelOffset(0.007,"XYZ")
        tStyle.SetLabelSize(0.04,"XYZ")
        tStyle.SetTitleFont(42,"XYZ")
        tStyle.SetTitleSize(0.047,"XYZ")
        tStyle.SetTitleXOffset(1.2)
        tStyle.SetTitleYOffset(1.7)

        #legend
        tStyle.SetLegendBorderSize(0)
        tStyle.SetLegendTextSize(self.legendTextSize)
        tStyle.SetLegendFont(42)

        #assign changes to gROOT current style
        tStyle.cd()

        return tStyle

    def __beautify__(self, canvas, CMSextraLabel, eraLabel):
        #################################
        #Add CMS and era labels to canvas
        #################################

        leftMargin = canvas.GetLeftMargin()
        rightMargin = canvas.GetRightMargin()
        topMargin = canvas.GetTopMargin() 
        canvas.cd() 

        #CMStext
        CMSlabel = "CMS"
        CMSextraOffset = 0.10  
        CMStext = ROOT.TLatex()
        CMSextra = ROOT.TLatex()

        CMStext.SetNDC()
        CMSextra.SetNDC()

        CMStext.SetTextAngle(0)
        CMSextra.SetTextAngle(0)

        CMStext.SetTextColor(ROOT.kBlack)
        CMSextra.SetTextColor(ROOT.kBlack) 
 
        CMStext.SetTextFont(61)
        CMSextra.SetTextFont(52)
  
        CMStext.SetTextAlign(11)
        CMStext.SetTextSize(0.045)
        CMSextra.SetTextSize(0.035)

        CMStext.DrawLatex(leftMargin,1.-topMargin+0.01,CMSlabel)
        CMSextra.DrawLatex(leftMargin+CMSextraOffset,1-topMargin+0.01,CMSextraLabel) 

        #Era text
        eraText = ROOT.TLatex()
        eraText.SetNDC()
        eraText.SetTextAngle(0)
        eraText.SetTextColor(ROOT.kBlack)
        eraText.SetTextFont(42)    
        eraText.SetTextAlign(33)
        eraText.SetTextSize(0.035)
        eraText.DrawLatex(1.-rightMargin,1.-topMargin+0.035,eraLabel)

        #Redraw axis
        canvas.RedrawAxis()  

    def __cleanSingle__(self):
        #####################
        #Move all final files 
        #to output directory
        #####################

        for dirpath,dirs,files in os.walk(self.cwd):
            if dirpath != self.cwd: continue
            for n_file in files:
                if ".png" in n_file or ".pdf" in n_file or ".eps" in n_file:
                    self.__log__("i","File "+n_file+" was created.")
                    os.system("mv "+n_file+" "+self.outputDir)
        self.__log__("i","Done.")

    def __clean__(self):
        #####################
        #Move all final files 
        #to output directory
        #####################
       
        for datafile in self.dataFiles:  
            os.system("mv "+datafile+" "+self.outputDir)
        for mcfile in self.MCobjects:
            os.system("mv "+mcfile+" "+self.outputDir)
        for dirpath,dirs,files in os.walk(self.cwd):
            if dirpath != self.cwd: continue
            for n_file in files:
                if ".png" in n_file or ".pdf" in n_file or ".eps" in n_file:
                    self.__log__("i","File "+n_file+" was created.")
                    os.system("mv "+n_file+" "+self.outputDir) 
        self.__log__("i","Done.")

    def __finalize__(self):
        ##########################################
        #List all created figures and say goodbye 
        ##########################################

        for dirpath,dirs,files in os.walk(self.outputDir):
            for n_file in files:
                if ".png" in n_file or ".pdf" in n_file or ".eps" in n_file:
                    self.__log__("i","File "+n_file+" was created.")
        self.__log__("i","Done.")        

        
    def addDATA(self,filename):
        #############################################################
        #Add DATA objects in one file to be plotted together with MC
        #############################################################
        if os.path.isfile(str(filename)):
            self.__log__("i","DATA file: "+str(filename)+" was added for plotting.")
            self.dataFiles.append(str(filename)) 
        elif os.path.isfile(os.path.join(str(filename),self.fileBaseName)):
            self.__log__("i","DATA file: "+os.path.join(str(filename),self.fileBaseName)+" was added for plotting.")
            self.dataFiles.append(os.path.join(str(filename),self.fileBaseName))
        else:
            self.__log__("w","DATA file: "+os.path.join(str(filename),self.fileBaseName)+" NOT found.")

    def addDirDATA(self, dataDir):
        #####################################################################
        #Add directory of single DATA files to be plotted together with MC
        #####################################################################
        if os.path.isdir(dataDir):
            self.__log__("i","DATA dir: "+dataDir+" was added for plotting.")
            self.dataDirs.append(dataDir)
        else: 
            self.__log__("w","DATA dir: "+dataDir+" NOT found.")

        #Create list of dataFiles #FIXME for multiple DATA inputs
        if len(self.dataDirs) != 0:
            if self.args['isDMR']:   
                for dataDir in self.dataDirs:
                    for root,dirs,files in os.walk(dataDir):
                        for dir in dirs:
                            if dir.startswith("offline"): 
                                self.dataFiles.append(dataDir+"/"+dir+"/ExtendedOfflineValidation_Images/OfflineValidationSummary.root")

    def addDirMC(self, mcDir):
        #####################################################################
        #Add directory of single MC file to be plotted together with DATA
        #####################################################################
        if os.path.isdir(mcDir):
            self.__log__("i","MC dir: "+mcDir+" was added for plotting.")
            nFiles = 0
            for dirpath,dirs,files in os.walk(mcDir):
                for file in files: 
                    if self.fileBaseName.replace(".root","") in file and file.endswith(".root"):
                        self.__log__("i","MC file: "+str(file)+" was added for plotting.")
                        self.MCobjects.append(os.path.join(dirpath,file))
                        nFiles += 1
            if nFiles == 0:
                self.__log__("w","No MC file found in "+str(mcDir)+".")
        else:
            self.__log__("w","MC dir: "+mcDir+" NOT found.")

    def addMC(self,filename):
        #############################################################
        #Add MC objects in one file to be plotted together with DATA
        #############################################################
        if os.path.isfile(str(filename)):
            self.__log__("i","MC file: "+str(filename)+" was added for plotting.")
            self.MCobjects.append(str(filename))
        elif os.path.isfile(os.path.join(str(filename),self.fileBaseName)):
            self.__log__("i","MC file: "+os.path.join(str(filename),self.fileBaseName)+" was added for plotting.") 
            self.MCobjects.append(os.path.join(str(filename),self.fileBaseName))
        else:
            self.__log__("w","MC file: "+str(os.path.join(str(filename),self.fileBaseName))+" NOT found.")  

    def plotSingle(self):
        ##############################################
	#Auxiliary plotter for unweighted Data and MC
        ##############################################

        #check for input file and create output dir
        self.__createSingleArchitecture__()

        #access histograms in rootfiles, select different validation objects and store them separately
        objects = self.__defineSingleObjects__()
        objectsData = objects['DATA']
        objectsMC = objects['MC'] 

        #set histogram style
        for objDict in objectsData:
            self.__setTHStyle__(objDict)
        for objDict in objectsMC:
            self.__setTHStyle__(objDict)

        #really plot
        ROOT.gROOT.SetBatch(True) #turn off printing canvas on screen
        ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 1001;") #turn off printing messages on terminal
      
        for objDict in objectsData:
            for segment in self.segments:
                for var in self.varsX:
                    id = "0"
                    for key in objDict.keys():
                        for _obj in objDict[key]:     
                            id = _obj['id']  
                    canvas = ROOT.TCanvas(id+"_"+var+"_"+segment)
                    canvas.cd()

                    #set labels positioning
                    segmentText = {'text' : segment, 'xmin' : 0.0, 'xmax' : 0.0}
                    statText = {'xmin' : 0.0, 'xmax' : 0.0}
                    if "median" in var:
                         segmentText['xmin'] = 2.5
                         segmentText['xmax'] = 3.5
                         statText['xmin'] = 0.20
                         statText['xmax'] = 0.85
                    else:
                         segmentText['xmin'] = 1.4
                         segmentText['xmax'] = 1.6
                         statText['xmin'] = 0.65
                         statText['xmax'] = 0.95

                    #order plots & prepare y-axis scale factors  
                    isEmpty = True
                    maxY = 0.0
                    objGroup = []
                    for objName in self.objNameList: #follow plotting order
                        for obj in objDict[objName]:
                            if obj['var'] == var and obj['segment'] == segment:
                                if obj['hist'].GetBinContent(obj['hist'].GetMaximumBin()) >= maxY:
                                    maxY = obj['hist'].GetBinContent(obj['hist'].GetMaximumBin())
                                isEmpty = False
                                legendLabel = objName.replace("_"," ")
                                if len(self.args['labels']) != 0:
                                    legendLabel = self.args['labels'][self.args['objects'].index(objName)]

                                #Add MC for each data file
                                histsMC = []
                                labelsMC = []
                                statsMC = []      
                                for objDictMC in objectsMC:
                                    for objNameMC in self.objNameListMC:
                                        for objMC in objDictMC[objNameMC]:
                                            if objMC['var'] == var and objMC['segment'] == segment:
                                                if objMC['hist'].GetBinContent(objMC['hist'].GetMaximumBin()) >= maxY:
                                                    maxY = objMC['hist'].GetBinContent(objMC['hist'].GetMaximumBin())
                                                legendLabelMC = objNameMC.replace("_"," ")
                                                if len(self.args['labels']) != 0:
                                                    legendLabelMC = self.args['labels'][self.args['objects'].index(objNameMC)]
                                                objMC['hist'].SetDirectory(0)
                                                histsMC.append(objMC['hist'])
                                                labelsMC.append(legendLabelMC)
                                                statsMC.append(self.__getStat__(objMC['hist'],var))
                                objGroup.append({'hist'    : obj['hist'],
                                                 'histsMC' : histsMC,
                                                 'labelsMC': labelsMC,
                                                 'statsMC' : statsMC, 
                                                 'label'   : legendLabel,
                                                 'stat'    : self.__getStat__(obj['hist'],var)
                                                })
                    #draw & save
                    if not isEmpty:
                        datasetType = "singlemuon" #FIXME make it an option
                        legMinY = (1./self.legendOffset)+(1.-1./self.legendOffset)*(self.maxEntriesPerColumn-len(objGroup))/(self.maxEntriesPerColumn*3)
                        nColumns = 1
                        if len(objGroup) > self.maxEntriesPerColumn:
                            nColumns = 2
                            legMinY = 1./self.legendOffset
                        leg = ROOT.TLegend(0.08,legMinY,0.45,0.88)
                        leg.SetNColumns(nColumns)
                        seg = ROOT.TLatex()
                        maxX = objGroup[0]['hist'].GetXaxis().GetXmax()
                        stat = self._styledTPaveText(maxX*statText['xmin'],(legMinY+0.025)*self.legendOffset*maxY,maxX*statText['xmax'],0.95*self.legendOffset*maxY,var)
                        for igroup,group in enumerate(objGroup):
                            group['hist'].GetYaxis().SetRangeUser(0,self.legendOffset*maxY)
                            leg.AddEntry(group['hist'],group['label'],"l")
                            stat.AddText(group['stat'])
                            group['hist'].Draw("HISTSAME")
                            #for last data group add also MC
                            if igroup == len(objGroup)-1:
                                for ihist,histmc in enumerate(group['histsMC']):
                                    leg.AddEntry(histmc,group['labelsMC'][ihist],"l")
                                    stat.AddText(group['statsMC'][ihist])
                                    histmc.Draw("HISTSAME")         
                        leg.Draw("SAME")
                        seg.DrawLatex(segmentText['xmin'],self.segmentTextOffset['ymin']*maxY,segmentText['text'])
                        stat.Draw("SAME")
                        self.__beautify__(canvas,self.args['CMSlabel'],self.args['Rlabel'])
                        canvas.SaveAs(self.outputDir+"/offline_"+datasetType+"_"+str(id)+"/"+var+"_"+segment+".png")
                        canvas.SaveAs(self.outputDir+"/offline_"+datasetType+"_"+str(id)+"/"+var+"_"+segment+".pdf")
                        self.__log__("i","Saving "+self.outputDir+"/offline_"+datasetType+"_"+str(id)+"/"+var+"_"+segment)

            #finalize
            #self.__cleanSingle__()       
            self.__log__("i","Done.") 

    def plot(self):
        ##################
        #Main plotter part
        ##################
       
        #check for input file and create output dir if needed 
        self.__createArchitecture__()

        #access histograms in rootfiles, select different validation objects and store them separately
        objects = self.__defineObjects__() 

        #set histogram style
        currentStyle = self.__setTHStyle__(objects) #NOTE: for CMSSW_11 and higher, currentStyle must be returned to plotting function

        #really plot
        ROOT.gROOT.SetBatch(True) #turn off printing canvas on screen
        ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = 1001;") #turn off printing messages on terminal
       
        for segment in self.segments:
            for var in self.varsX:
                canvas = ROOT.TCanvas(var+"_"+segment)
                canvas.cd()

                #set labels positioning
                segmentText = {'text' : segment, 'xmin' : 0.0, 'xmax' : 0.0}
                statText = {'xmin' : 0.0, 'xmax' : 0.0}
                if "median" in var:
                     segmentText['xmin'] = 2.5
                     segmentText['xmax'] = 3.5
                     statText['xmin'] = 0.27
                     statText['xmax'] = 0.92
                else:
                     segmentText['xmin'] = 1.4
                     segmentText['xmax'] = 1.6
                     statText['xmin'] = 0.75
                     statText['xmax'] = 0.95
                
                #order plots & prepare y-axis scale factors  
                isEmpty = True
                maxY = 0.0
                objGroup = [] 
                for objName in self.objNameList: #follow plotting order
                    for obj in objects[objName]:
                        if obj['var'] == var and obj['segment'] == segment:
                            if obj['hist'].GetBinContent(obj['hist'].GetMaximumBin()) >= maxY:
                                maxY = obj['hist'].GetBinContent(obj['hist'].GetMaximumBin())
                            isEmpty = False
                            legendLabel = objName.replace("_"," ")
                            if len(self.args['labels']) != 0:
                                legendLabel = self.args['labels'][self.args['objects'].index(objName)]
                            drawStyle = ""
                            legStyle = ""
                            if obj['type'] == "MC": 
                                drawStyle = "HIST SAME"
                                legStyle = "l"
                            if obj['type'] == "DATA": 
                                drawStyle += "P HIST SAME"
                                legStyle = "p"     
                            objGroup.append({'hist'        : obj['hist'], 
                                             'label'       : legendLabel,
                                             'stat'        : self.__getStat__(obj['hist'],var),
                                             'drawStyle'   : drawStyle,
                                             'legStyle'    : legStyle  
                                            })

                #draw & save
                if not isEmpty:
                    legMinY = (1./self.legendOffset)+(1.-1./self.legendOffset)*(self.maxEntriesPerColumn-len(objGroup))/(self.maxEntriesPerColumn*3)
                    nColumns = 1   
                    if len(objGroup) > self.maxEntriesPerColumn:
                        nColumns = 2
                        legMinY = 1./self.legendOffset
                    leg = ROOT.TLegend(0.20,legMinY,0.50,0.88)
                    leg.SetNColumns(nColumns)
                    seg = ROOT.TLatex()
                    maxX = objGroup[0]['hist'].GetXaxis().GetXmax() 
                    stat = self._styledTPaveText(maxX*statText['xmin'],(legMinY+0.025)*self.legendOffset*maxY,maxX*statText['xmax'],0.95*self.legendOffset*maxY,var)
                    for group in objGroup:
                        group['hist'].GetYaxis().SetRangeUser(0,self.legendOffset*maxY)
                        leg.AddEntry(group['hist'],group['label'],group['legStyle'])
                        stat.AddText(group['stat']) 
                        group['hist'].Draw(group['drawStyle'])   
                    leg.Draw("SAME")  
                    seg.DrawLatex(segmentText['xmin'],self.segmentTextOffset['ymin']*maxY,segmentText['text'])
                    stat.Draw("SAME")
                    self.__beautify__(canvas,self.args['CMSlabel'],self.args['Rlabel'])
                    canvas.Print(self.outputDir+"/"+var+"_"+segment+".png")
                    canvas.Print(self.outputDir+"/"+var+"_"+segment+".pdf") 
        
        #finalize
        self.__finalize__()
