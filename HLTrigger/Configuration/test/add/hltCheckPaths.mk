# Makefile for hltCheckPaths
# 
# This makefile checks that each trigger path gives the same results if run stand-alone or in the global trigger table.
# Given a configName, it extracts from ConfDB the global table (Global_Table.py), the list of path, and a configuration 
# for each path (TriggerName.py).
# Then both the trigger table and each paths are run, the corresponding TrigReport is extracted from the log, and compared.
# 
# run "make help" to get synopsis.
#
# Version 1.4.0, 2009.08.19
# Andrea Bocci <andrea.bocci@cern.ch>

# TODO - cleanup
# - define a function or variable in place of all the 'echo -e "$(CLEAR)..."'

.PHONY: all list check clean summary help

.SECONDEXPANSION:

# configuration goes here
PROCESS := HLT
EVENTS  := 100

# supported menus
MENUS := 8E29 1E31 GRun

HLT_GRun_CONFIG     := /dev/CMSSW_3_3_0/backport/GRun
HLT_GRun_GLOBALTAG  := STARTUP31X_V8
HLT_GRun_SOURCE     := file:RelVal_DigiL1Raw_8E29.root

HLT_8E29_CONFIG     := /dev/CMSSW_3_3_0/backport/8E29
HLT_8E29_GLOBALTAG  := STARTUP31X_V8
HLT_8E29_SOURCE     := file:RelVal_DigiL1Raw_8E29.root

HLT_1E31_CONFIG     := /dev/CMSSW_3_3_0/backport/1E31
HLT_1E31_GLOBALTAG  := MC_31X_V9
HLT_1E31_SOURCE     := file:RelVal_DigiL1Raw_1E31.root

# more configuration, useful to debug the Makefile itself
CMSRUN    := cmsRun
GETCONFIG := hltConfigFromDB

# check for cmsRun environmnt
ifeq (,$(CMSSW_RELEASE_BASE))
  $(error Please configure the cmsRun environment with the 'cmsenv' command)
endif

# internal stuff
NORMAL  := \033[0m
BOLD    := \033[1m
RED     := \033[31m
GREEN   := \033[32m
YELLOW  := \033[33m
BLUE    := \033[34m
COLUMN  := \033[100G
CLEAR   := \033[0K
DONE    := $(CLEAR)$(COLUMN)$(GREEN)done$(NORMAL)
WARNING := $(CLEAR)$(COLUMN)$(YELLOW)warning$(NORMAL)
ERROR   := $(CLEAR)$(COLUMN)$(RED)error$(NORMAL)

# assume all relevant targets are of the form LUMI_NAME[.TYPE]
LUMI = $(strip $(word 1, $(subst _, , $@)) )
NAME = $(strip $(subst $(LUMI)_, , $(word 1, $(subst ., , $@))) )
TYPE = $(strip $(word 2, $(subst ., , $@)) )

LIST_OF_GRun_PATHS := $(shell hltConfigFromDB --configName $(HLT_GRun_CONFIG) --nopsets --noedsources --noes --noservices --nooutput --nosequences --nomodules --format python | gawk '/^process\..*(AlCa|HLT)_.* = cms.Path/ { print gensub(/^process\.(.*(AlCa|HLT)_.*) = cms.Path.*/, "\\1", 1) }' | sort)
LIST_OF_GRun_PYS   := $(patsubst %, GRun_%.py,   $(LIST_OF_GRun_PATHS))
LIST_OF_GRun_LOGS  := $(patsubst %, GRun_%.log,  $(LIST_OF_GRun_PATHS))
LIST_OF_GRun_DIFFS := $(patsubst %, GRun_%.diff, $(LIST_OF_GRun_PATHS))

LIST_OF_8E29_PATHS := $(shell hltConfigFromDB --configName $(HLT_8E29_CONFIG) --nopsets --noedsources --noes --noservices --nooutput --nosequences --nomodules --format python | gawk '/^process\..*(AlCa|HLT)_.* = cms.Path/ { print gensub(/^process\.(.*(AlCa|HLT)_.*) = cms.Path.*/, "\\1", 1) }' | sort)
LIST_OF_8E29_PYS   := $(patsubst %, 8E29_%.py,   $(LIST_OF_8E29_PATHS))
LIST_OF_8E29_LOGS  := $(patsubst %, 8E29_%.log,  $(LIST_OF_8E29_PATHS))
LIST_OF_8E29_DIFFS := $(patsubst %, 8E29_%.diff, $(LIST_OF_8E29_PATHS))

LIST_OF_1E31_PATHS := $(shell hltConfigFromDB --configName $(HLT_1E31_CONFIG) --nopsets --noedsources --noes --noservices --nooutput --nosequences --nomodules --format python | gawk '/^process\..*(AlCa|HLT)_.* = cms.Path/ { print gensub(/^process\.(.*(AlCa|HLT)_.*) = cms.Path.*/, "\\1", 1) }' | sort)
LIST_OF_1E31_PYS   := $(patsubst %, 1E31_%.py,   $(LIST_OF_1E31_PATHS))
LIST_OF_1E31_LOGS  := $(patsubst %, 1E31_%.log,  $(LIST_OF_1E31_PATHS))
LIST_OF_1E31_DIFFS := $(patsubst %, 1E31_%.diff, $(LIST_OF_1E31_PATHS))

TABLE_PATHS   := $(patsubst %, %_GlobalTable, $(MENUS))
TABLE_PYS     := $(patsubst %, %.py,  $(TABLE_PATHS))
TABLE_LOGS    := $(patsubst %, %.log, $(TABLE_PATHS))

LIST_OF_PATHS := $(sort $(LIST_OF_8E29_PATHS) $(LIST_OF_1E31_PATHS) $(LIST_OF_GRun_PATHS))
LIST_OF_PYS   := $(LIST_OF_8E29_PYS)   $(LIST_OF_1E31_PYS)   $(LIST_OF_GRun_PYS)
LIST_OF_LOGS  := $(LIST_OF_8E29_LOGS)  $(LIST_OF_1E31_LOGS)  $(LIST_OF_GRun_LOGS)
LIST_OF_DIFFS := $(LIST_OF_8E29_DIFFS) $(LIST_OF_1E31_DIFFS) $(LIST_OF_GRun_DIFFS)

.PHONY: $(LIST_OF_PATHS)

# do not delete these
.SECONDARY: $(LIST_OF_LOGS)

all:    pys logs check summary

pys:    $(TABLE_PYS)  $(LIST_OF_PYS)

logs:   $(TABLE_LOGS) $(LIST_OF_LOGS)

check:  $(LIST_OF_DIFFS)

clean:
	@rm -f .database_* *.py *.pyc *.log *.single *.master *.diff
  
list: 
	@echo "$(LIST_OF_PATHS)"

summary: | check
	@echo
	@DIFF=`find $(LIST_OF_DIFFS) -not -empty | sed -e's/\.diff$$//'`; \
	if [ -z "$$DIFF" ]; then \
	  echo "No discrepancies found."; \
	else \
	  echo "Found discrepancies in the trigger paths:" ;\
	  for P in $$DIFF; do echo -e "\t$(RED)$$P$(NORMAL)$(CLEAR)"; done; \
	fi
	@DIFF=`find $(LIST_OF_DIFFS) -not -empty | sed -e's/\.diff$$//'`; \
	PARTIAL=$$(gawk 'FNR==1 { NAME=gensub(/^...._(.*)\.log/,"\\1",1,FILENAME); LUMI=gensub(/^(....)_.*\.log/,"\\1",1,FILENAME); HEADER="TrigReport ---------- Modules in Path: "NAME" ------------"; } $$0 ~ HEADER { while ($$0 !~ /TrigReport.*hltBoolEnd/) getline; if ($$5 == 0) printf "%s_%s\n", LUMI, NAME; nextfile; }' *_*.log); \
	PARTIAL=$$(for P in $$PARTIAL; do echo $$DIFF | grep -q $$P || echo $$P; done); \
	if [ -n "$$PARTIAL" ]; then \
	  echo; \
	  echo "These paths where not fully excercised:"; \
	  for P in $$PARTIAL; do echo -e "\t$(YELLOW)$$P$(NORMAL)$(CLEAR)"; done; \
	  echo; \
	fi
	@MISS=`grep -L 'TrigReport.*hltBoolEnd' $(LIST_OF_LOGS) | sed -e's/\.log$$//'`; \
	if [ -n "$$MISS" ]; then \
	  echo; \
	  echo "These paths miss the trailing hltBoolEnd filter:"; \
	  for P in $$MISS; do echo -e "\t$(YELLOW)$$P$(NORMAL)$(CLEAR)"; done; \
	  echo; \
	fi

# these are kinda tricky: we need rules based on the file content, not on its modification date
DB_GRun_NEW:=$(shell echo -e "CONFIG=$(HLT_GRun_CONFIG)\nSOURCE=$(HLT_GRun_SOURCE)\nGLOBALTAG=$(HLT_GRun_GLOBALTAG)" | md5sum | cut -c -32)
DB_GRun_SUM:=$(shell [ -f .database_GRun ] && cat .database_GRun | md5sum | cut -c -32)

ifneq ($(DB_GRun_NEW), $(DB_GRun_SUM))
.database_GRun:
	@echo -e "CONFIG=$(HLT_GRun_CONFIG)\nSOURCE=$(HLT_GRun_SOURCE)\nGLOBALTAG=$(HLT_GRun_GLOBALTAG)" > .database_GRun
endif

DB_8E29_NEW:=$(shell echo -e "CONFIG=$(HLT_8E29_CONFIG)\nSOURCE=$(HLT_8E29_SOURCE)\nGLOBALTAG=$(HLT_8E29_GLOBALTAG)" | md5sum | cut -c -32)
DB_8E29_SUM:=$(shell [ -f .database_8E29 ] && cat .database_8E29 | md5sum | cut -c -32)

ifneq ($(DB_8E29_NEW), $(DB_8E29_SUM))
.database_8E29:
	@echo -e "CONFIG=$(HLT_8E29_CONFIG)\nSOURCE=$(HLT_8E29_SOURCE)\nGLOBALTAG=$(HLT_8E29_GLOBALTAG)" > .database_8E29
endif

DB_1E31_NEW:=$(shell echo -e "CONFIG=$(HLT_1E31_CONFIG)\nSOURCE=$(HLT_1E31_SOURCE)\nGLOBALTAG=$(HLT_1E31_GLOBALTAG)" | md5sum | cut -c -32)
DB_1E31_SUM:=$(shell [ -f .database_1E31 ] && cat .database_1E31 | md5sum | cut -c -32)

ifneq ($(DB_1E31_NEW), $(DB_1E31_SUM))
.database_1E31:
	@echo -e "CONFIG=$(HLT_1E31_CONFIG)\nSOURCE=$(HLT_1E31_SOURCE)\nGLOBALTAG=$(HLT_1E31_GLOBALTAG)" > .database_1E31
endif
# end of tricky rules

# rules to write the python configurations
$(TABLE_PYS): .database_$$(LUMI)
	@echo -e "ConfDB [$(BLUE)$(HLT_$(LUMI)_CONFIG)$(NORMAL)] menu $(BOLD)$(LUMI)_GlobalTable$(NORMAL)$(CLEAR)"
	@$(GETCONFIG) --configName $(HLT_$(LUMI)_CONFIG) --input $(HLT_$(LUMI)_SOURCE) --nopsets --nooutput --services -PrescaleService --esmodules -l1GtTriggerMenuXml,-L1GtTriggerMaskAlgoTrigTrivialProducer --format python --paths -OfflineOutput                                | sed -e's/^process = cms.Process(.*)/process = cms.Process( "$(PROCESS)" )/' -e's/^process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32( $(EVENTS) ) )/process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32( 100 ) )/' > $(LUMI)_GlobalTable.py
	@sed -e '/^process.streams/,/^)/d' -e'/^process.datasets/,/^)/d'         -i $(LUMI)_GlobalTable.py
	@sed -e 's/cms.InputTag( "source" )/cms.InputTag( "rawDataCollector" )/' -i $(LUMI)_GlobalTable.py
	@sed -e 's/cms.string( "source" )/cms.string( "rawDataCollector" )/'     -i $(LUMI)_GlobalTable.py
	@sed -e '/DTUnpackingModule/a\ \ \ \ inputLabel = cms.untracked.InputTag( "rawDataCollector" ),' -i $(LUMI)_GlobalTable.py
	@echo -e "process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_CONDITIONS'"                   >> $(LUMI)_GlobalTable.py
	@echo -e "process.GlobalTag.globaltag = '$(HLT_$(LUMI)_GLOBALTAG)'"                                 >> $(LUMI)_GlobalTable.py
	@echo -e "process.options = cms.untracked.PSet(\n    wantSummary = cms.untracked.bool( True )\n)\n" >> $(LUMI)_GlobalTable.py

$(LIST_OF_PYS): .database_$$(LUMI)
	@echo -e "ConfDB [$(BLUE)$(HLT_$(LUMI)_CONFIG)$(NORMAL)] path $(BOLD)$(NAME)$(NORMAL)$(CLEAR)"
	@$(GETCONFIG) --configName $(HLT_$(LUMI)_CONFIG) --input $(HLT_$(LUMI)_SOURCE) --nopsets --nooutput --services -PrescaleService --esmodules -l1GtTriggerMenuXml,-L1GtTriggerMaskAlgoTrigTrivialProducer --format python --paths HLTriggerFirstPath,$(NAME),HLTriggerFinalPath | sed -e's/^process = cms.Process(.*)/process = cms.Process( "$(PROCESS)" )/' -e's/^process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32( $(EVENTS) ) )/process.maxEvents = cms.untracked.PSet(  input = cms.untracked.int32( 100 ) )/' > $@
	@sed -e '/^process.streams/,/^)/d' -e'/^process.datasets/,/^)/d'         -i $@
	@sed -e 's/cms.InputTag( "source" )/cms.InputTag( "rawDataCollector" )/' -i $@
	@sed -e 's/cms.string( "source" )/cms.string( "rawDataCollector" )/'     -i $@
	@sed -e '/DTUnpackingModule/a\ \ \ \ inputLabel = cms.untracked.InputTag( "rawDataCollector" ),' -i $@
	@echo -e "process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_CONDITIONS'"                   >> $@
	@echo -e "process.GlobalTag.globaltag = '$(HLT_$(LUMI)_GLOBALTAG)'"                                 >> $@
	@echo -e "process.options = cms.untracked.PSet(\n    wantSummary = cms.untracked.bool( True )\n)\n" >> $@

# rules to run cmsRun and produce log files
$(TABLE_LOGS): %.log: %.py
	@echo -e -n "running $(LUMI) Global Table$(CLEAR)\r"
	@$(CMSRUN) $< >& $@
	@echo -e "running $(LUMI) Global Table$(DONE)"

$(LIST_OF_LOGS): %.log: %.py
	@echo -e -n "running $(LUMI) trigger path $(NAME)$(CLEAR)\r"
	@$(CMSRUN) $< >& $@
	@echo -e "running $(LUMI) trigger path $(NAME)$(DONE)"

# rules to extract the output fragments from GlobalTable and single triggers log files
$(LIST_OF_DIFFS:diff=single): %.single: %.log
	@cat $<                      | gawk 'BEGIN { FOUND=0 } /TrigReport ---------- Modules in Path: $(NAME) ------------/ { FOUND=1; print; getline; print "TrigReport  Trig    Visited     Passed     Failed      Error Name"; next; } /^$$/ {FOUND=0; next; } // { if (FOUND) printf "TrigReport   %3d %10d %10d %10d %10d %s\n",$$2,$$4,$$5,$$6,$$7,$$8 }' > $@

$(LIST_OF_DIFFS:diff=master): %.master: $$(LUMI)_GlobalTable.log
	@cat $(LUMI)_GlobalTable.log | gawk 'BEGIN { FOUND=0 } /TrigReport ---------- Modules in Path: $(NAME) ------------/ { FOUND=1; print; getline; print "TrigReport  Trig    Visited     Passed     Failed      Error Name"; next; } /^$$/ {FOUND=0; next; } // { if (FOUND) printf "TrigReport   %3d %10d %10d %10d %10d %s\n",$$2,$$4,$$5,$$6,$$7,$$8 }' > $@

# rules to compute the diff of the fragments, and check the results
$(LIST_OF_DIFFS): %.diff: %.log $$(LUMI)_GlobalTable.log | %.single %.master
	@echo -e -n "checking $(LUMI) trigger path $(NAME)$(CLEAR)\r"
	@if diff -w -U 999 $*.master $*.single > $@; then \
	    PARTIAL=$$(gawk ' \
	      /TrigReport ---------- Modules in Path: $(NAME) ------------/ \
	      { \
	        while ($$0 !~ /TrigReport.*hltBoolEnd/)  \
	          if (getline <= 0) { \
	            print "$*";  \
	            exit; \
	          } \
	          if ($$5 == 0) \
	            print "$*";  \
	          exit;  \
	      } \
	    ' $<); \
	    if [ -z "$$PARTIAL" ]; then \
	      echo -e "checking $(LUMI) trigger path $(NAME)$(DONE)"; \
	    else \
	      echo -e "checking $(LUMI) trigger path $(NAME)$(WARNING)"; \
	    fi \
	else \
	    echo -e "checking $(LUMI) trigger path $(NAME)$(ERROR)"; \
	fi

# help
help:
	@echo 'This makefile checks that each trigger path gives the same results if run stand-alone or in the global trigger table.'
	@echo 'Given a configName, it extracts from ConfDB the global table (*.py_GlobalTable), the list of path, and a configuration'
	@echo 'for each path (TriggerName.py).'
	@echo 'Then both the trigger table and each paths are run, the corresponding TrigReport is extracted from the log, and compared.'
	@echo
	@echo 'After running each path, a warning is issued if a path is not fully excersided, i.e. no events pass all the filters.'
	@echo 'After comparing with the global table, an error is issued if there are discrepenaices in the TrigReport (at any level).'
	@echo
	@echo 'Working files:'
	@echo '  .database_LUMI         running conditions: ConfDB configuration, input dataset, GlobalTag'
	@echo '  LUMI_GlobalTable.py    configuration for the global table'
	@echo '  LUMI_GlobalTable.log   output of cmsRun LUMI_GlobalTable.py'
	@echo '  TriggerName.py         configuration for TriggerName path'
	@echo '  TriggerName.log        output of cmsRun TriggerName.py'
	@echo '  TriggerName.diff       differences between TrigReport for TriggerName between global table run and stand-alone run'
	@echo
	@echo 'Supported targets:'
	@echo '  all                    same as "pys logs diffs summary"'
	@echo '  list                   list all trigger paths'
	@echo '  pys                    extract all (python) configuration files'
	@echo '  logs                   run all configuration files'
	@echo '  check                  extract all differences'
	@echo '  summary                analyze all *available* diff and log files to print the list of trigger paths with discrepancies and/or with no accepted events'
	@echo '  help                   print a simple description of this tool'
	@echo
	@echo '  TriggerName            run the full chain for trigger TriggerName: extract TriggerName.py, cmsRun logging to TriggerName.log,'
	@echo '                         if necessary extract and run the global table, and compare the TrigResults.'
	@echo
	@echo 'Version 1.4.0, 2009.08.19'
	@echo 'Andrea Bocci <andrea.bocci@cern.ch>'
