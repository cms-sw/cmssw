
DIR=hpstanc

# Directories
TRANS=${DIR}/transform
DB=${DIR}/db
LOG=${DIR}/log
EVAL=${DIR}/eval
XML=${DIR}/xml

all: ${EVAL}/eval.pdf 

clean:
	rm -f ${DB}/*
	rm -f ${LOG}/*
	rm -f ${TRANS}/*

# Combine the signal and background evaluation
${EVAL}/eval.pdf: make_eval_plots.py ${EVAL}/eval_signal.root ${EVAL}/eval_background.root
	./make_eval_plots.py $@ ${EVAL}/eval_signal.root ${EVAL}/eval_background.root

# Evaluate the signal
${EVAL}/eval_signal.root: evaluate_cfg.py ${DB}/computers.db ${TRANS}/transforms.py signalfiles.test
	./$< inputFiles_load=signalfiles.test signal=1 db=${DB}/computers.db transform=${TRANS}/transforms.py outputFile=$@

# Evaluate the background
${EVAL}/eval_background.root: evaluate_cfg.py ${DB}/computers.db ${TRANS}/transforms.py backgroundfiles.test
	./$< inputFiles_load=backgroundfiles.test signal=0 db=${DB}/computers.db transform=${TRANS}/transforms.py outputFile=$@

#################################################################
#  MVA transformation
#################################################################

# Merge all of the indifividual transforms into a single file that can be read
# by RecoTauMVATransform
${TRANS}/transforms.py: ${TRANS}/transform_1prong0pi0.py ${TRANS}/transform_1prong1pi0.py ${TRANS}/transform_1prong2pi0.py ${TRANS}/transform_3prong0pi0.py 
	./mergeTransforms.py $@ $^

# Compute transforms for the individual decay modes
${TRANS}/transform_%.py: ${TRANS}/signal_transform_%.root ${TRANS}/background_transform_%.root computeTransform.py
	./computeTransform.py -s ${TRANS}/signal_transform_$*.root \
	  -b ${TRANS}/background_transform_$*.root -o $@

${TRANS}/signal_transform_%.root: ${DB}/computers.db evaluateMode_cfg.py signalfiles.test
	./evaluateMode_cfg.py inputFiles_load=signalfiles.test db=$< outputFile=$@  \
	  signal=1 \
	  tracks=`echo $* | sed "s|[^0-9]*\([0-9]\)prong\([0-9]\)pi0|\1|"` \
	  pizeros=`echo $* | sed "s|[^0-9]*\([0-9]\)prong\([0-9]\)pi0|\2|"` 

${TRANS}/background_transform_%.root: ${DB}/computers.db evaluateMode_cfg.py backgroundfiles.test
	./evaluateMode_cfg.py inputFiles_load=backgroundfiles.test db=$< outputFile=$@ \
	  signal=0 \
	  tracks=`echo $* | sed "s|[^0-9]*\([0-9]\)prong\([0-9]\)pi0|\1|"` \
	  pizeros=`echo $* | sed "s|[^0-9]*\([0-9]\)prong\([0-9]\)pi0|\2|"` 
	
#################################################################
#  Database management - depends on trained sub-mvas
#################################################################

.PRECIOUS: ${DB}/1prong0pi0.db ${DB}/1prong1pi0.db ${DB}/1prong2pi0.db ${DB}/3prong0pi0.db 

.SECONDARY:

# Merge the computers together
${DB}/computers.db: ${DB}/1prong0pi0.mva ${DB}/1prong1pi0.mva ${DB}/1prong2pi0.mva ${DB}/3prong0pi0.mva 
	rm -f $@
	./merge_dbs.py $@ $^

# Switch formats
${DB}/%.mva: ${DB}/%.db
	./dump_db.py $^

#################################################################
#  Running the actual training - depends on MVA definition and samples
#################################################################
${DB}/%.db: ${XML}/%.xml signalfiles.list backgroundfiles.list trainMVA_cfg.py
	rm -f ${LOG}/${@F}.log
	nice cmsRun trainMVA_cfg.py xml=$< \
	  tracks=`echo $< | sed "s|[^0-9]*\([0-9]\)prong\([0-9]\)pi0.*.xml|\1|"` \
	  pizeros=`echo $< | sed "s|[^0-9]*\([0-9]\)prong\([0-9]\)pi0.*.xml|\2|"` \
	  inputFiles_load=signalfiles.list \
	  inputFiles_load=backgroundfiles.list \
	  outputFile=${DB}/$(@F).temp.db >& ${LOG}/$(@F).log
	mv ${DB}/$(@F).temp.db $@

#################################################################
#  Defining sample lists
#################################################################

signalfiles.list:
	ls /data1/friis/TauMVATraining2/Ztautau/signal_training*.root \
	  | tail -n 100 \
	  | awk 'NR % 3 != 0' | sed "s|^|file:|" >> $@

backgroundfiles.list:
	ls /data1/friis/TauMVATraining2/Background_Run2010A/background_training*.root \
	  | tail -n 100 \
	  | awk 'NR % 3 != 0' | sed "s|^|file:|" >> $@
	ls /data1/friis/TauMVATraining2/Background_Run2010B/background_training*.root \
	  | tail -n 100  \
	  | awk 'NR % 3 != 0' | sed "s|^|file:|" >> $@

# unbiased testing datasets
signalfiles.test:
	ls /data1/friis/TauMVATraining2/Ztautau/signal_training*.root \
	  | awk 'NR % 3 == 0' | sed "s|^|file:|" >> $@

# FIXME
backgroundfiles.test:
	ls /data1/friis/TauMVATraining2/Background_Run2010A/background_training*.root \
	  | awk 'NR % 3 == 0' | sed "s|^|file:|" >> $@
	ls /data1/friis/TauMVATraining2/Background_Run2010B/background_training*.root \
	  | awk 'NR % 3 == 0' | sed "s|^|file:|" >> $@

#################################################################
#  Skimming information
#################################################################

# Converting EPS to PDF so ROOT behaves
%.pdf: %.ps
	epstopdf $<

# Plot skim efficiency plots
plots/skim_plots.ps: skim_plots.py signal_skim_plots.root background_skim_plots.root
	mkdir -p plots
	./skim_plots.py signal_skim_plots.root background_skim_plots.root

# Merge the skim plot root files together
%_skim_plots.root: %files.list
	# Conver the list of data files into the associated skim plot file and
	# add them together using xargs --> hadd
	cat $< | sed "s|training|skim_plots|" | xargs hadd -f $@
