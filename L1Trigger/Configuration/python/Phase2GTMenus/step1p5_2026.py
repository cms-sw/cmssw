# P2GT implementation of the step 1.5 menu based on Run 3 (2025 v1_3_0)
# Subset of the v0p0_2026 menu: the step 1.5 seed files only

_seeds = "L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.v0p0_2026"

menu = [
    f"{_seeds}.l1tGTMenu_lepSeeds_cff",
    f"{_seeds}.l1tGTMenu_hadr_metSeeds_cff",
    f"{_seeds}.l1tGTMenu_crossLepSeeds_cff",
    f"{_seeds}.l1tGTMenu_hadr_crossLepSeeds_cff",
    f"{_seeds}.l1tGTMenu_vbfSeeds_cff",
    f"{_seeds}.l1tGTMenu_displacedSeeds_cff",
]
