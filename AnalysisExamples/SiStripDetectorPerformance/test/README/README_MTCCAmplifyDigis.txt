SUMMARY
    Config starts only one module that would amplify SiStripDigis and store
  new values as a separate collection in ROOT tuple. At the moment ALL SiStrip 
  digis are amplified (all available in Event) according to formula:
    
    NewAdc = ScaleFactor * Gauss( OldAdc, Sigma);

  where:
    - OldAdc       ADC of original (old) DIGI
    - Sigma        self descriptive. At the moment only two values are used: 
                   one for TOB and one for TIB.

                   [Note: values are in ADC counts]
                   [Note: values are double, e.g. '2.7']
    - ScaleFactor  Some additional constant for scaling. Four values are used:
                   2 values for TOB and 2 values for TIB (only 4 layers are 
                   used at all)
                   [Note: values are double, e.g. '.959']

  [Note: THE CODE WAS TESTED ONLY FOR MONTE-CARLO... Real Data tests are coming
         soon and changes will be posted.]

CONTACT INFO
  Samvel Khalatian (samvel at fnal dot gov)

INSTRUCTION
  
  Simple Run
  ----------
    cmsRun MTCCAmplifyDigis_default.cfg

  Advanced Run
  ------------
    1. Put next four lines after include of MTCCAmplifyDigis module in order
       to use user defined values of sigmas in Amplification:

         replace modMTCCAmplifyDigis.oDigiAmplifySigmas = {
           untracked double dTIB = <PUT_HERE_NEW_VALUE_FOR_TIB>
           untracked double dTOB = <PUT_HERE_NEW_VALUE_FOR_TOB>
         }

    2. Put next lines after include of MTCCAmplifyDigis module in order to
       use user defined values of scaling: 

         replace modMTCCAmplifyDigis.oDigiScaleFactors = {
           untracked PSet oTIB = {
             untracked double dL1 = <PUT_HERE_NEW_VALUE_FOR_TIB_LAYER_1>
             untracked double dL2 = <PUT_HERE_NEW_VALUE_FOR_TIB_LAYER_2>
           }

           untracked PSet oTOB = {
             untracked double dL1 = <PUT_HERE_NEW_VALUE_FOR_TOB_LAYER_1>
             untracked double dL2 = <PUT_HERE_NEW_VALUE_FOR_TOB_LAYER_2>
           }
         }

    3. In case there are several Digis collection in file or Label/ProdInstName
       of Digi changed put next two lines after include of MTCCAmplifyDigis
       module:
        
         replace modMTCCAmplifyDigis.oSiStripDigisLabel        = 
           "<PUT_HERE_NEW_LABEL>"
         replace modMTCCAmplifyDigis.oSiStripDigisProdInstName = 
           "<PUT_HERE_NEW_PROD_INST_NAME>"

    4. To change produced Digis collection Label put next line after include of
       MTCCAmplifyDigis module:
         
         replace modMTCCAmplifyDigis.oNewSiStripDigisLabel     = 
           "<PUT_HERE_NEW_LABEL>"
