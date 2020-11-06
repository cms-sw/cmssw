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
        everything_tree_dict[target_path] = DQMItem(layout=[simple_details])
    return layout_function

# Create the actual layout function
# Set your "base path" here:
info_layout = make_basic_layout_function(dqmitems, "Info/Layouts/")

# Define the actual layouts:

info_layout("Info/EventInfo/reportSummaryMap",
            "1 - High Voltage (HV) per LumiSection",
            "High Voltage (HV) flag for all sub-detector parts, per LumiSection (LS), based on information from DCS, injected into the events by SCAL.")

info_layout("Info/EventInfo/ProcessedLS",
            "2 - Processed LumiSections",
            "Processing flag per LumiSection (LS): -1 means that the LS was not processed, +1 means that the LS was processed.")

info_layout("Info/ProvInfo/runIsComplete",
            "3 - Run is completely processed",
            "This flag becomes 1 if the run is completely processed.<br><br><ul><li>For <b>Prompt Reconstruction</b> this is the case by default, since the info is only uploaded to the GUI when the run is completely processed.</li><li>However for <b>Express Reconstruction</b>, the GUI will show intermediate results and the run you're looking at might not be completely processed yet.</li></ul>")

info_layout("Info/ProvInfo/CMSSW",
            "4 - Version of CMSSW used",
            "Version of CMSSW used to process this run.")

info_layout("Info/CMSSWInfo/globalTag_Step1",
            "5 - Global Tag used for filling",
            "Global Tag (GT) used for <i>step_1</i>, i.e. the LumiSection based filling of the plots.")

info_layout("Info/CMSSWInfo/globalTag_Harvesting",
            "6 - Global Tag used for harvesting",
            "Global Tag (GT) used for the <i>harvesting</i> step, i.e. when the statistics for all the lumisections are combined into the final plots.")
