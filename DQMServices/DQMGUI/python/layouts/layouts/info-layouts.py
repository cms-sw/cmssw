from .adapt_to_new_backend import *
dqmitems={}

# Definition of custom function used further below
# (It's unlikely that you need to modify this code)
def make_basic_layout_function(everything_tree_dict, base_path):

    # Prepare a layout function that takes the following 3 arguments:
    #   source_path: The path in "everything" where the plot is now
    #   new_relative_path: The path relative to your base path of your layouts
    #                      where you want the layout to appear
    #   description: Description of the layout, to be displayed in the GUI when
    #                the "Describe" button is clicked. Note that this
    #                description can use basic html syntax.

    def layout_function(source_path, new_relative_path, description):
        target_path = base_path + new_relative_path
        simple_details = [{"path":source_path, "description": description}]
        everything_tree_dict[target_path] = [simple_details]
    return layout_function

# Create the actual layout function
# Set your "base path" here:
info_layout = make_basic_layout_function(dqmitems, "Info/Layouts/")

# Define the actual layouts:

info_layout("Info/EventInfo/reportSummaryMap",
            "1 - HV and Beam Status per LumiSection",
            "High Voltage (HV) flag for all sub-detector parts, as well as basic beam information, per LumiSection (LS), based on information from DCS, injected into the events by SCAL.")

info_layout("Info/ProvInfo/Run Type",
            "2 - Run key set for DQM",
            "Run key set for the DQM module in the DAQ Level0 Funtionmanager. The value typically defines whether the run is <i>collisions</i> or <i>cosmics</i>. Depending on this run key the DQM applications running for the different sub-systems can process the run in different ways.")

info_layout("Info/ProvInfo/hltKey",
            "3 - HLT menu used",
            "Path of the HLT menu that was used for this run.")

info_layout("Info/ProvInfo/CMSSW",
            "4 - Version of CMSSW used",
            "Version of CMSSW used to process this run. There might be extra PRs added manually. Please refer to the <a href='https://twiki.cern.ch/twiki/bin/view/CMS/DQMP5TagCollector'>Tag Collector</a> for details.")

info_layout("Info/ProvInfo/Globaltag",
            "5 - Global Tag used",
            "Global Tag (GT) used in the online DQM cluster.")

apply_dqm_items_to_new_back_end(dqmitems, __file__)
