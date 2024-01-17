from rich.console import Console

caloSettingsConsole = Console()

try:
    from L1Trigger.L1TCalorimeter.caloParams.customiseSettings import *
except ModuleNotFoundError:
    caloSettingsConsole.print_exception()
    caloSettingsConsole.print("Could not find the caloParams configurations. They have been moved to a separate repository:")
    caloSettingsConsole.print("https://github.com/cms-l1t-offline/caloParams")
    caloSettingsConsole.print("Please find the caloParams in L1TOfflineSoftware repository")
    caloSettingsConsole.print("And clone them to L1Trigger/L1TCalorimeter/python/")
    caloSettingsConsole.print("Once present, all functions may be imported from here (for legacy reasons), or the caloParams/customiseSettings.py")
    exit(1)