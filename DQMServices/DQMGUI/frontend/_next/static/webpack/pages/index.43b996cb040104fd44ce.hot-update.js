webpackHotUpdate_N_E("pages/index",{

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _styles_theme__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../styles/theme */ "./styles/theme.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/workspaces/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];












var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__["store"]),
      workspace = _React$useContext.workspace,
      setWorkspace = _React$useContext.setWorkspace;

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var initialWorkspace = _config_config__WEBPACK_IMPORTED_MODULE_13__["functions_config"].mode === 'ONLINE' ? workspaces[0].workspaces[1].label : workspaces[0].workspaces[3].label;
  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    setWorkspace(initialWorkspace);
    return function () {
      return setWorkspace(initialWorkspace);
    };
  }, []);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1]; // make a workspace set from context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: [__jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledButton"], {
      color: _styles_theme__WEBPACK_IMPORTED_MODULE_12__["theme"].colors.secondary.main,
      background: "white",
      key: "Close",
      onClick: function onClick() {
        return toggleWorkspaces(false);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 57,
        columnNumber: 13
      }
    }, "Close")],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 67,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 69,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 71,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "9wsb3E7mFlyFmQpi1Uvfk2BcVak=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL2luZGV4LnRzeCJdLCJuYW1lcyI6WyJUYWJQYW5lIiwiVGFicyIsIldvcmtzcGFjZXMiLCJSZWFjdCIsInN0b3JlIiwid29ya3NwYWNlIiwic2V0V29ya3NwYWNlIiwid29ya3NwYWNlcyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwib25saW5lV29ya3NwYWNlIiwib2ZmbGluZVdvcnNrcGFjZSIsImluaXRpYWxXb3Jrc3BhY2UiLCJsYWJlbCIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5Iiwib3BlbldvcmtzcGFjZXMiLCJ0b2dnbGVXb3Jrc3BhY2VzIiwidGhlbWUiLCJjb2xvcnMiLCJzZWNvbmRhcnkiLCJtYWluIiwibWFwIiwic3ViV29ya3NwYWNlIiwic2V0V29ya3NwYWNlVG9RdWVyeSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0lBRVFBLE8sR0FBWUMseUMsQ0FBWkQsTzs7QUFNUixJQUFNRSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQUEsMEJBQ2FDLGdEQUFBLENBQWlCQyxnRUFBakIsQ0FEYjtBQUFBLE1BQ2ZDLFNBRGUscUJBQ2ZBLFNBRGU7QUFBQSxNQUNKQyxZQURJLHFCQUNKQSxZQURJOztBQUd2QixNQUFNQyxVQUFVLEdBQ2RDLGdFQUFnQixDQUFDQyxJQUFqQixLQUEwQixRQUExQixHQUFxQ0MsNkRBQXJDLEdBQXVEQyw4REFEekQ7QUFHQSxNQUFNQyxnQkFBZ0IsR0FBR0osZ0VBQWdCLENBQUNDLElBQWpCLEtBQTBCLFFBQTFCLEdBQXFDRixVQUFVLENBQUMsQ0FBRCxDQUFWLENBQWNBLFVBQWQsQ0FBeUIsQ0FBekIsRUFBNEJNLEtBQWpFLEdBQXlFTixVQUFVLENBQUMsQ0FB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S0FiWGIsVTtBQXdFU0EseUVBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNDNiOTk2Y2IwNDAxMDRmZDQ0Y2UuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IFRhYnMsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyB3b3Jrc3BhY2VzIGFzIG9mZmxpbmVXb3Jza3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29mZmxpbmUnO1xuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvbmxpbmVXb3Jrc3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29ubGluZSc7XG5pbXBvcnQgeyBTdHlsZWRNb2RhbCB9IGZyb20gJy4uL3ZpZXdEZXRhaWxzTWVudS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCBGb3JtIGZyb20gJ2FudGQvbGliL2Zvcm0vRm9ybSc7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkQnV0dG9uIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBzZXRXb3Jrc3BhY2VUb1F1ZXJ5IH0gZnJvbSAnLi91dGlscyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgdGhlbWUgfSBmcm9tICcuLi8uLi9zdHlsZXMvdGhlbWUnO1xuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xuXG5jb25zdCB7IFRhYlBhbmUgfSA9IFRhYnM7XG5cbmludGVyZmFjZSBXb3JzcGFjZVByb3BzIHtcbiAgbGFiZWw6IHN0cmluZztcbiAgd29ya3NwYWNlczogYW55O1xufVxuY29uc3QgV29ya3NwYWNlcyA9ICgpID0+IHtcbiAgY29uc3QgeyB3b3Jrc3BhY2UsIHNldFdvcmtzcGFjZSB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcblxuICBjb25zdCB3b3Jrc3BhY2VzID1cbiAgICBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gb25saW5lV29ya3NwYWNlIDogb2ZmbGluZVdvcnNrcGFjZTtcbiAgICBcbiAgY29uc3QgaW5pdGlhbFdvcmtzcGFjZSA9IGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScgPyB3b3Jrc3BhY2VzWzBdLndvcmtzcGFjZXNbMV0ubGFiZWwgOiB3b3Jrc3BhY2VzWzBdLndvcmtzcGFjZXNbM10ubGFiZWxcblxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xuICAgIHNldFdvcmtzcGFjZShpbml0aWFsV29ya3NwYWNlKVxuICAgIHJldHVybiAoKSA9PiBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcbiAgfSwgW10pXG5cbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IFtvcGVuV29ya3NwYWNlcywgdG9nZ2xlV29ya3NwYWNlc10gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XG5cbiAgLy8gbWFrZSBhIHdvcmtzcGFjZSBzZXQgZnJvbSBjb250ZXh0XG4gIHJldHVybiAoXG4gICAgPEZvcm0+XG4gICAgICA8U3R5bGVkRm9ybUl0ZW0gbGFiZWxjb2xvcj1cIndoaXRlXCIgbGFiZWw9XCJXb3Jrc3BhY2VcIj5cbiAgICAgICAgPEJ1dHRvblxuICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcbiAgICAgICAgICAgIHRvZ2dsZVdvcmtzcGFjZXMoIW9wZW5Xb3Jrc3BhY2VzKTtcbiAgICAgICAgICB9fVxuICAgICAgICAgIHR5cGU9XCJsaW5rXCJcbiAgICAgICAgPlxuICAgICAgICAgIHt3b3Jrc3BhY2V9XG4gICAgICAgIDwvQnV0dG9uPlxuICAgICAgICA8U3R5bGVkTW9kYWxcbiAgICAgICAgICB0aXRsZT1cIldvcmtzcGFjZXNcIlxuICAgICAgICAgIHZpc2libGU9e29wZW5Xb3Jrc3BhY2VzfVxuICAgICAgICAgIG9uQ2FuY2VsPXsoKSA9PiB0b2dnbGVXb3Jrc3BhY2VzKGZhbHNlKX1cbiAgICAgICAgICBmb290ZXI9e1tcbiAgICAgICAgICAgIDxTdHlsZWRCdXR0b25cbiAgICAgICAgICAgICAgY29sb3I9e3RoZW1lLmNvbG9ycy5zZWNvbmRhcnkubWFpbn1cbiAgICAgICAgICAgICAgYmFja2dyb3VuZD1cIndoaXRlXCJcbiAgICAgICAgICAgICAga2V5PVwiQ2xvc2VcIlxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB0b2dnbGVXb3Jrc3BhY2VzKGZhbHNlKX1cbiAgICAgICAgICAgID5cbiAgICAgICAgICAgICAgQ2xvc2VcbiAgICAgICAgICAgIDwvU3R5bGVkQnV0dG9uPixcbiAgICAgICAgICBdfVxuICAgICAgICA+XG4gICAgICAgICAgPFRhYnMgZGVmYXVsdEFjdGl2ZUtleT1cIjFcIiB0eXBlPVwiY2FyZFwiPlxuICAgICAgICAgICAge3dvcmtzcGFjZXMubWFwKCh3b3Jrc3BhY2U6IFdvcnNwYWNlUHJvcHMpID0+IChcbiAgICAgICAgICAgICAgPFRhYlBhbmUga2V5PXt3b3Jrc3BhY2UubGFiZWx9IHRhYj17d29ya3NwYWNlLmxhYmVsfT5cbiAgICAgICAgICAgICAgICB7d29ya3NwYWNlLndvcmtzcGFjZXMubWFwKChzdWJXb3Jrc3BhY2U6IGFueSkgPT4gKFxuICAgICAgICAgICAgICAgICAgPEJ1dHRvblxuICAgICAgICAgICAgICAgICAgICBrZXk9e3N1YldvcmtzcGFjZS5sYWJlbH1cbiAgICAgICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxuICAgICAgICAgICAgICAgICAgICBvbkNsaWNrPXthc3luYyAoKSA9PiB7XG4gICAgICAgICAgICAgICAgICAgICAgc2V0V29ya3NwYWNlKHN1YldvcmtzcGFjZS5sYWJlbCk7XG4gICAgICAgICAgICAgICAgICAgICAgdG9nZ2xlV29ya3NwYWNlcyghb3BlbldvcmtzcGFjZXMpO1xuICAgICAgICAgICAgICAgICAgICAgIC8vaWYgd29ya3NwYWNlIGlzIHNlbGVjdGVkLCBmb2xkZXJfcGF0aCBpbiBxdWVyeSBpcyBzZXQgdG8gJycuIFRoZW4gd2UgY2FuIHJlZ29uaXplXG4gICAgICAgICAgICAgICAgICAgICAgLy90aGF0IHdvcmtzcGFjZSBpcyBzZWxlY3RlZCwgYW5kIHdlZSBuZWVkIHRvIGZpbHRlciB0aGUgZm9yc3QgbGF5ZXIgb2YgZm9sZGVycy5cbiAgICAgICAgICAgICAgICAgICAgICBhd2FpdCBzZXRXb3Jrc3BhY2VUb1F1ZXJ5KHF1ZXJ5LCBzdWJXb3Jrc3BhY2UubGFiZWwpO1xuICAgICAgICAgICAgICAgICAgICB9fVxuICAgICAgICAgICAgICAgICAgPlxuICAgICAgICAgICAgICAgICAgICB7c3ViV29ya3NwYWNlLmxhYmVsfVxuICAgICAgICAgICAgICAgICAgPC9CdXR0b24+XG4gICAgICAgICAgICAgICAgKSl9XG4gICAgICAgICAgICAgIDwvVGFiUGFuZT5cbiAgICAgICAgICAgICkpfVxuICAgICAgICAgIDwvVGFicz5cbiAgICAgICAgPC9TdHlsZWRNb2RhbD5cbiAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgPC9Gb3JtPlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgV29ya3NwYWNlcztcbiJdLCJzb3VyY2VSb290IjoiIn0=